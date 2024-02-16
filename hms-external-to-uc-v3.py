# Databricks notebook source
# MAGIC %md
# MAGIC # External tables to UC managed tables
# MAGIC
# MAGIC This notebook will migrate tables from a Hive metastore to a UC catalog.
# MAGIC
# MAGIC **Important:**
# MAGIC - This notebook needs to run in a cluster with that supports Unity Catalog. Also, it needs to be attached to an instance profile that gives read/write access to Glue. Write access is necessary because the UC SYNC commands creates a property (information purpose) in the tables being migrated
# MAGIC - All external locations related to the tables in scope must registered as Unity Catalog External Locations. Also, the user running this script and creating tables must have CREATE TABLE permission on the External Locations
# MAGIC - The principal running this job must have CREATE EXTERNAL TABLE access to the Unity Catalog External Locations
# MAGIC - The Spark Property "spark.databricks.sync.command.enableManagedTable" must be set to True in the Cluster Level to allow the migration of MANAGED Tables with MANAGED External Locations
# MAGIC - To make sure that the SYNC statement is not updating the existing delta table (by adding properties to it), set spark.databricks.sync.command.disableSourceTableWrites to True
# MAGIC
# MAGIC **Input Parameters:**
# MAGIC - destination_catalog: UC Catalog where the tables will be created
# MAGIC - is_dry_run: Indicates whether the job will execute a dry run
# MAGIC - hms_db_to_migrate: The database in hive_metastore to be migrated to UC. This parameter takes precedence over "table_inventory_path". Therefore, if both parameters are provided, this script will consider only "hms_db_to_migrate"
# MAGIC - table_inventory_path: Path to the CSV file containing the list of tables to be migrated. This file must be a CSV with the columns "db" and "table". The script automatically assume that these tables belong to the hive_metastore catalog
# MAGIC - table_owner_uc: The principal who should own the tables in UC after migration. The script will change the table ownership to this user at the end of the process
# MAGIC
# MAGIC **Query Results:**
# MAGIC
# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

#dbutils.widgets.removeAll()
#dbutils.widgets.text("destination_catalog", "")
#dbutils.widgets.text("is_dry_run", "True")
#dbutils.widgets.text("table_inventory_path", "")
#dbutils.widgets.text("hms_db_to_migrate", "")
#dbutils.widgets.text("table_owner_uc", "")

# COMMAND ----------

from pyspark.sql.functions import *
from concurrent.futures import ThreadPoolExecutor
import itertools 
import os
import datetime
import re

# Source and target information
source_catalog = "hive_metastore"
destination_catalog = dbutils.widgets.get("destination_catalog")

table_inventory_path = dbutils.widgets.get("table_inventory_path")
hms_database_to_migrate = dbutils.widgets.get("hms_db_to_migrate")

# Safety check
if destination_catalog.strip() == "hive_metastore" or not destination_catalog.strip():
    raise Exception("ERROR: Invalid destionation catalog.") 

# Set table Owner in UC
table_owner = dbutils.widgets.get("table_owner_uc")
if not table_owner.strip():
    raise Exception("ERROR: UC table owner must be specified.") 

# Log Database
log_database = "uc_migration_log_db"

if hms_database_to_migrate.strip():
    log_sync_tables = "uc_migration_log_tables_tb_" + hms_database_to_migrate
    log_scan = "uc_migration_log_scan_" + hms_database_to_migrate

# Parallelism when syncing the tables
#default_parallelism = os.cpu_count()

# Is this a dry run?
is_dry_run = dbutils.widgets.get("is_dry_run")
dry_run = True
if is_dry_run.upper() == "FALSE":
    dry_run = False

# COMMAND ----------

# MAGIC %md
# MAGIC # PART I: Create Log Tables

# COMMAND ----------

# Setting up log table
log_table_name = f"{destination_catalog}.{log_database}.{log_sync_tables}"
log_scan_table_name = f"{destination_catalog}.{log_database}.{log_scan}"
spark.sql(f"CREATE DATABASE IF NOT EXISTS`{destination_catalog}`.{log_database}")  
spark.sql(f"CREATE TABLE  IF NOT EXISTS {log_table_name} (source_table STRING, target_table STRING, table_location STRING, table_provider STRING, command STRING, status STRING, description STRING, dry_run BOOLEAN, batch_id LONG, execution_time TIMESTAMP)")
spark.sql(f"CREATE TABLE  IF NOT EXISTS {log_scan_table_name} (db_name STRING, table_name STRING, table_type STRING, table_location STRING, table_provider STRING, batch_id LONG, error STRING)")

# Generating batch id
batch_id = spark.sql(f"select max(batch_id) + 1 from {log_table_name}").collect()[0][0]

if not batch_id:
    batch_id = 1

# Cleaning log table for the batch id
spark.sql(f"DELETE FROM {log_scan_table_name} WHERE batch_id = {batch_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC # PART II: Create Inventory of HMS tables to migrate

# COMMAND ----------

# Build a list of tables to be migrated

if hms_database_to_migrate.strip():
    list_dbs = hms_database_to_migrate.split(",")
    list_tables = spark.sql(f"show tables in hive_metastore.{list_dbs[0]}").drop("isTemporary").withColumnRenamed("tableName", "table").withColumnRenamed("database", "db")
    if len(list_dbs) > 1:
        for db in list_dbs[1:]:
            list_tables = list_tables.union(spark.sql(f"show tables in hive_metastore.{db}").drop("isTemporary").withColumnRenamed("tableName", "table").withColumnRenamed("database", "db"))
elif table_inventory_path.strip():
    list_tables = spark.read.option("header", True).csv(table_inventory_path)
else:
    raise Exception("ERROR: A database or the path of a CSV file containing the list of tables to migrate must be provided.")
    
list_tables = list_tables.withColumn("hms_table", concat(lit("hive_metastore."), col("db"), lit("."), col("table"))).select("hms_table")
display(list_tables)

# COMMAND ----------

# MAGIC %md
# MAGIC # PART III: Collect Metadata

# COMMAND ----------

def scan_tables(inventory_obj, batch_id):
  
  descriptions = []
  database_name = ""
  table_name = ""
  table_type = ""
  table_location = ""
  table_provider = ""
  table_full_name = inventory_obj.hms_table

  try:

    desc = spark.sql(f"DESCRIBE FORMATTED {table_full_name}").filter("col_name = 'Location' OR col_name='Database' OR col_name='Table' OR col_name='Type' OR col_name='Provider'")
    
    for info in desc.collect():
      if info["col_name"] == "Database":
        database_name = info["data_type"]
      elif info["col_name"] == "Table":
        table_name = info["data_type"]
      elif info["col_name"] == "Type":
        table_type = info["data_type"]
      elif info["col_name"] == "Location":
        table_location = info["data_type"]
      elif info["col_name"] == "Provider":
        table_provider = info["data_type"]

    descriptions.append((database_name, table_name, table_type, table_location, table_provider, batch_id, ""))
  except Exception as ex:
    error = str(ex).replace("'", "\\'")
    descriptions.append(("", table_full_name, "", "", "", batch_id, error))

  # Save result in a log table
  try:

    scan_table_log = spark.createDataFrame(descriptions, ['db_name', 'table_name', 'table_type', 'table_location', 'table_provider', 'batch_id', 'error'])

    scan_table_log.write.format("delta").mode("append").saveAsTable(f"{destination_catalog}.{log_database}.{log_scan}")

  except Exception as e:
    error = str(e).replace("'", "\\'")
    print(f"ERROR: Unable to update the log table {destination_catalog}.{log_database}.{log_scan}: {error}")

# COMMAND ----------

# Executing scan command in a thread pull
with ThreadPoolExecutor(max_workers = 4) as executor:
    executor.map(scan_tables, list_tables.collect(), itertools.repeat(batch_id))

# COMMAND ----------

# Optimizing log table
spark.sql(f"OPTIMIZE {log_scan_table_name}")

# COMMAND ----------

# Tables that we were able to scan. Only these tables will be considered for migration
source_catalog_tables_valid = spark.sql(f"select * from {log_scan_table_name} where batch_id = {batch_id}").where(col("error") == "")
display(source_catalog_tables_valid)

# COMMAND ----------

# Tables that we were not able to scan. These tables will not be considered for migration
source_catalog_tables_invalid = spark.sql(f"select * from {log_scan_table_name} where batch_id = {batch_id}").where(col("error") != "")
display(source_catalog_tables_invalid)

# COMMAND ----------

# MAGIC %md
# MAGIC # PART IV: Upgrade tables to UC

# COMMAND ----------

def sync_tables(inventory_obj, batch_id, dry_run, execution_time):

  db_name = inventory_obj["db_name"]
  tb_name = inventory_obj["table_name"]
  tb_location = inventory_obj["table_location"]
  tb_provider = inventory_obj["table_provider"]
  full_name_source = f"{source_catalog}.{db_name}.{tb_name}"
  full_name_target = f"{destination_catalog}.{db_name}.{tb_name}"
  command = ""
  status_code = ""
  descriptions = ""
  log = []

  try:

    # Create external table in Unity Catalog
    if tb_provider == "hive":
      if not dry_run:
        spark.sql(f"DROP TABLE IF EXISTS {full_name_target}")
        creat_statement = spark.sql(f'show create table {full_name_source}').collect()[0][0]
        command = re.sub("CREATE TABLE .* \(", f"CREATE TABLE {full_name_target} (", creat_statement)
        spark.sql(command).collect()
        status_code = "SUCCESS"
        descriptions = f"Table {full_name_target} created"
    else:
      if dry_run:
        command = f"SYNC table {full_name_target} from {full_name_source} dry run"
      else:
        command = f"SYNC table {full_name_target} from {full_name_source}"
      result = spark.sql(command).collect()[0]
      status_code = result["status_code"]
      descriptions = result["description"]

    # Generating log
    log.append((full_name_source, full_name_target, tb_location, tb_provider, command, status_code, descriptions, dry_run, batch_id, execution_time))

  except Exception as e:
    error = str(e).replace("'", "\\'")
    print(f"Unable to create table {full_name_source} to {full_name_target} of type {tb_provider}: {error}")
    # Generating log
    log.append((full_name_source, full_name_target, tb_location, tb_provider, command, "ERROR", error, dry_run, batch_id, execution_time))

  # Save result in a log table
  try:

    source_catalog_tables = spark.createDataFrame(log, ['source_table', 'target_table', 'table_location', 'table_provider', 'command', 'status', 'description', 'dry_run', 'batch_id', 'execution_time'])

    source_catalog_tables.write.format("delta").mode("append").saveAsTable(f"{destination_catalog}.{log_database}.{log_sync_tables}")

  except Exception as e:
    error = str(e).replace("'", "\\'")
    print(f"ERROR: Unable to update the log table {destination_catalog}.{log_database}.{log_sync_tables}: {error}")

# COMMAND ----------

# Sync tables to UC

execution_time = datetime.datetime.now()

# Executing Sync command in a thread pull
with ThreadPoolExecutor(max_workers = 8) as executor:
    executor.map(sync_tables, source_catalog_tables_valid.collect(), itertools.repeat(batch_id), itertools.repeat(dry_run), itertools.repeat(execution_time))

# COMMAND ----------

# Optimizing log table
spark.sql(f"OPTIMIZE {log_table_name}")

# COMMAND ----------

display(spark.sql(f"select * from {log_table_name} where batch_id = {batch_id}"))

# COMMAND ----------

#  Change table ownership in UC

tables_migrated = spark.sql(f"select target_table from {log_table_name} where batch_id = {batch_id} and dry_run = false").collect()

for item in tables_migrated:
    table_name = item.target_table
    try:
        spark.sql(f"ALTER TABLE {table_name} OWNER TO `{table_owner}`;")
    except Exception as e:
        error = str(e).replace("'", "\\'")
        print(f"Unable to change ownership of table {table_name} to {table_owner}: {error}")

# COMMAND ----------


