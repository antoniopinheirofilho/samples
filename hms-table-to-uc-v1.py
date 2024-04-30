# Databricks notebook source
# MAGIC %md
# MAGIC # External tables to UC managed tables
# MAGIC
# MAGIC This notebook will migrate tables from a Hive metastore to a UC catalog.
# MAGIC
# MAGIC **Important:**
# MAGIC - This notebook needs to run in a cluster with that supports Unity Catalog. It needs to have access to the existing HMS tables (via instance profile or cluster configuration)
# MAGIC - All external locations related to the tables in scope must be registered as Unity Catalog External Locations. Also, the user running this script and creating tables must have CREATE TABLE permission on the External Locations
# MAGIC - The principal running this job must have CREATE EXTERNAL TABLE access to the Unity Catalog External Locations
# MAGIC - The Spark Property "spark.databricks.sync.command.enableManagedTable" must be set to True in the Cluster Level to allow the migration of MANAGED Tables with MANAGED External Locations
# MAGIC - By default, the SYNC statement adds a property to the existing delta table. To prevent SYNC from adding this property, set spark.databricks.sync.command.disableSourceTableWrites to True
# MAGIC
# MAGIC **Input Parameters:**
# MAGIC - destination_catalog: UC Catalog where the tables will be created
# MAGIC - is_dry_run: Indicates whether the job will execute a dry run
# MAGIC - hms_db_to_migrate: The database in hive_metastore to be migrated to UC. This parameter takes precedence over "table_inventory_path". Therefore, if both parameters are provided, this script will consider only "hms_db_to_migrate"
# MAGIC - table_inventory_path: Path to the CSV file containing the list of tables to be migrated. This file must be a CSV with the columns "db" and "table". The script automatically assume that these tables belong to the hive_metastore catalog
# MAGIC - table_owner_uc: The principal who should own the tables in UC after migration. The script will change the table ownership to this user at the end of the process

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

#dbutils.widgets.removeAll()
#dbutils.widgets.text("destination_catalog", "dev")
#dbutils.widgets.text("is_dry_run", "True")
#dbutils.widgets.text("table_inventory_path", "")
#dbutils.widgets.text("hms_db_to_migrate", "")
#dbutils.widgets.text("table_owner_uc", "")

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC pip install sqlglot

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


table_inventory_path = ""
try:
    table_inventory_path = dbutils.widgets.get("table_inventory_path")
except:
    pass

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
log_sync_tables = "uc_migration_log_tables_tb"
log_scan = "uc_migration_log_scan"

# Parallelism when syncing the tables
#default_parallelism = os.cpu_count()

# Is this a dry run?
is_dry_run = dbutils.widgets.get("is_dry_run")
dry_run = True
if is_dry_run.upper() == "FALSE":
    dry_run = False

# COMMAND ----------

# MAGIC %md
# MAGIC # PART 1: Create Log Tables

# COMMAND ----------

# Setting up log table
log_table_name = f"{destination_catalog}.{log_database}.{log_sync_tables}"
log_scan_table_name = f"{destination_catalog}.{log_database}.{log_scan}"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {destination_catalog}.{log_database}")  
spark.sql(f"CREATE TABLE  IF NOT EXISTS {log_table_name} (source_table STRING, target_table STRING, table_location STRING, table_type STRING, table_provider STRING, command STRING, status STRING, description STRING, dry_run BOOLEAN, batch_id LONG, execution_time TIMESTAMP)")
spark.sql(f"CREATE TABLE  IF NOT EXISTS {log_scan_table_name} (db_name STRING, table_name STRING, table_type STRING, table_location STRING, table_provider STRING, batch_id LONG, error STRING)")

# Generating batch id
batch_id = spark.sql(f"select max(batch_id) + 1 from {log_table_name}").collect()[0][0]

if not batch_id:
    batch_id = 1

# Cleaning log table for the batch id
spark.sql(f"DELETE FROM {log_scan_table_name} WHERE batch_id = {batch_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC # PART 2: Create Inventory of HMS tables to migrate

# COMMAND ----------

# Build a list of tables to be migrated

if hms_database_to_migrate.strip():
    list_dbs = hms_database_to_migrate.split(",")
    list_tables = spark.sql(f"show tables in hive_metastore.{list_dbs[0]}").drop("isTemporary").withColumnRenamed("tableName", "table").withColumnRenamed("database", "db")
    if len(list_dbs) > 1:
        for db in list_dbs[1:]:
            try:
                list_tables = list_tables.union(spark.sql(f"show tables in hive_metastore.{db}").drop("isTemporary").withColumnRenamed("tableName", "table").withColumnRenamed("database", "db"))
            except Exception as e:
                print(e)
                pass
elif table_inventory_path.strip():
    list_tables = spark.read.option("header", True).csv(table_inventory_path)
else:
    raise Exception("ERROR: A database or the path of a CSV file containing the list of tables to migrate must be provided.")

list_schemas_df = list_tables.select("db").distinct()
list_tables = list_tables.withColumn("hms_table", concat(lit("hive_metastore."), col("db"), lit("."), col("table"))).select("hms_table")

display(list_tables)

# COMMAND ----------

# MAGIC %md
# MAGIC # PART 3: Collect Metadata

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
with ThreadPoolExecutor(max_workers = 8) as executor:
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
# MAGIC # PART 4: Migrate tables to UC

# COMMAND ----------

def sync_tables(inventory_obj, batch_id, dry_run, execution_time):

  db_name = inventory_obj["db_name"]
  tb_name = inventory_obj["table_name"]
  tb_location = inventory_obj["table_location"]
  tb_provider = inventory_obj["table_provider"]
  tb_type = inventory_obj["table_type"]
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
    log.append((full_name_source, full_name_target, tb_location, tb_provider, tb_type, command, status_code, descriptions, dry_run, batch_id, execution_time))

  except Exception as e:
    error = str(e).replace("'", "\\'")
    print(f"Unable to create table {full_name_source} to {full_name_target} of type {tb_provider}: {error}")
    # Generating log
    log.append((full_name_source, full_name_target, tb_location, tb_provider, tb_type, command, "ERROR", error, dry_run, batch_id, execution_time))

  # Save result in a log table
  try:

    source_catalog_tables = spark.createDataFrame(log, ['source_table', 'target_table', 'table_location', 'table_provider', 'table_type', 'command', 'status', 'description', 'dry_run', 'batch_id', 'execution_time'])

    source_catalog_tables.write.format("delta").mode("append").saveAsTable(f"{destination_catalog}.{log_database}.{log_sync_tables}")

  except Exception as e:
    error = str(e).replace("'", "\\'")
    print(f"ERROR: Unable to update the log table {destination_catalog}.{log_database}.{log_sync_tables}: {error}")

# COMMAND ----------

def migrate_managed_tables(inventory_obj, batch_id, dry_run, execution_time):

  db_name = inventory_obj["db_name"]
  tb_name = inventory_obj["table_name"]
  tb_location = inventory_obj["table_location"]
  tb_provider = inventory_obj["table_provider"]
  tb_type = inventory_obj["table_type"]
  full_name_source = f"{source_catalog}.{db_name}.{tb_name}"
  full_name_target = f"{destination_catalog}.{db_name}.{tb_name}"
  command = ""
  status_code = ""
  descriptions = ""
  log = []

  try:

    # Create external table in Unity Catalog
    if tb_provider == "delta":
      command = f"CREATE OR REPLACE TABLE {full_name_target} DEEP CLONE {full_name_source}"
      spark.sql(command)
      status_code = "SUCCESS"
      descriptions = f"Table {full_name_target} created"
    else:
      status_code = "ERROR"
      descriptions = f"Migration of managed {tb_provider} tables is not supported"

    # Generating log
    log.append((full_name_source, full_name_target, tb_location, tb_provider, tb_type, command, status_code, descriptions, dry_run, batch_id, execution_time))

  except Exception as e:
    error = str(e).replace("'", "\\'")
    print(f"Unable to create table {full_name_source} to {full_name_target} of type {tb_provider}: {error}")
    # Generating log
    log.append((full_name_source, full_name_target, tb_location, tb_provider, tb_type, command, "ERROR", error, dry_run, batch_id, execution_time))

  # Save result in a log table
  try:

    source_catalog_tables = spark.createDataFrame(log, ['source_table', 'target_table', 'table_location', 'table_provider', 'table_type', 'command', 'status', 'description', 'dry_run', 'batch_id', 'execution_time'])

    source_catalog_tables.write.format("delta").mode("append").saveAsTable(f"{destination_catalog}.{log_database}.{log_sync_tables}")

  except Exception as e:
    error = str(e).replace("'", "\\'")
    print(f"ERROR: Unable to update the log table {destination_catalog}.{log_database}.{log_sync_tables}: {error}")

# COMMAND ----------

import re
from sqlglot import parse_one, exp

def create_list_view_dependencies(view_name, list_view_dependencies):
  
  full_name_source = f"{source_catalog}.{view_name}"

  old_create_table = spark.sql(f"show create table {full_name_source}").collect()[0][0]

  if str.upper("AS WITH") in str.upper(old_create_table):
    old_create_table = f"CREATE VIEW {full_name_source} " + re.sub(r'^.*?(?=AS WITH)', '', old_create_table, flags=re.IGNORECASE | re.DOTALL)
  elif str.upper("AS SELECT") in str.upper(old_create_table):
    old_create_table = f"CREATE VIEW {full_name_source} " + re.sub(r'^.*?(?=AS SELECT)', '', old_create_table, flags=re.IGNORECASE | re.DOTALL)
  else:
    old_create_table = f"CREATE VIEW {full_name_source} " + re.sub(r'^.*?(?=AS \()', '', old_create_table, flags=re.IGNORECASE | re.DOTALL)

  v_dependency = []

  try:
    for table in parse_one(old_create_table, dialect="spark").find_all(exp.Table):
      db_dependency_name = table.db
      tb_dependency_name = table.name
      if db_dependency_name and tb_dependency_name:
        full_dependency_name = f"{db_dependency_name}.{tb_dependency_name}"
        if full_dependency_name != view_name:
          v_dependency.append(full_dependency_name)
  except:
    pass

  list_view_dependencies[view_name] = v_dependency

def dep_resolve(view_name, view_dependencies, view_mapping, resolved, seen):
   
  seen.append(view_name)

  for dependency in view_dependencies:
    if dependency not in resolved:
       
      if dependency in seen:
        raise Exception(f"Circular dependency detected. View name: {view_name}, Dependency name: {dependency}")
       
      if dependency in view_mapping:
        dep_resolve(dependency, view_mapping[dependency], view_mapping, resolved, seen)
      else:
        dep_resolve(dependency, [], view_mapping, resolved, seen)

  resolved.append(view_name)
  seen.remove(view_name)

def migrate_views(view_name, batch_id, dry_run, execution_time):

  full_name_source = f"{source_catalog}.{view_name}"
  full_name_target = f"{destination_catalog}.{view_name}"
  command = ""
  status_code = ""
  descriptions = ""
  log = []

  try:

    old_create_table = spark.sql(f"show create table {full_name_source}").collect()[0][0]

    if str.upper("AS WITH") in str.upper(old_create_table):
      old_create_table = f"CREATE VIEW {view_name} " + re.sub(r'^.*?(?=AS WITH)', '', old_create_table, flags=re.IGNORECASE | re.DOTALL)
    elif str.upper("AS SELECT") in str.upper(old_create_table):
      old_create_table = f"CREATE VIEW {view_name} " + re.sub(r'^.*?(?=AS SELECT)', '', old_create_table, flags=re.IGNORECASE | re.DOTALL)
    else:
      old_create_table = f"CREATE VIEW {view_name} " + re.sub(r'^.*?(?=AS \()', '', old_create_table, flags=re.IGNORECASE | re.DOTALL)

    list_dependencies = []
    for table in parse_one(old_create_table, dialect="spark").find_all(exp.Table):
      dp_db_name = table.db
      dp_tb_name = table.name
      if dp_db_name and dp_tb_name:
        list_dependencies.append(f"{dp_db_name}.{dp_tb_name}")

    list_dependencies = list(set(list_dependencies))

    new_create_table = old_create_table
    for full_tb_name in list_dependencies:
      full_uc_table_name = f"{destination_catalog}.{full_tb_name}"
      new_create_table = re.sub(r'\b{}\b'.format(re.escape(full_tb_name)), full_uc_table_name, new_create_table)

    new_create_table = new_create_table.replace("CREATE VIEW", "CREATE OR REPLACE VIEW")
    print(new_create_table)
    # Safety check
    if new_create_table.startswith(f"CREATE OR REPLACE VIEW {destination_catalog}."):
      command = new_create_table
      spark.sql(command)
      status_code = "SUCCESS"
      descriptions = f"View {full_name_target} created"
    else:
      raise Exception("DDL malformatted.")

    # Generating log
    log.append((full_name_source, full_name_target, "", "", "VIEW", command, status_code, descriptions, dry_run, batch_id, execution_time))

  except Exception as e:
    error = str(e).replace("'", "\\'")
    print(f"Unable to create view {full_name_target} from {full_name_source}: {error}")
    # Generating log
    log.append((full_name_source, full_name_target, "", "", "VIEW", command, "ERROR", error, dry_run, batch_id, execution_time))

  # Save result in a log table
  try:

    source_catalog_tables = spark.createDataFrame(log, ['source_table', 'target_table', 'table_location', 'table_provider', 'table_type', 'command', 'status', 'description', 'dry_run', 'batch_id', 'execution_time'])

    source_catalog_tables.write.format("delta").mode("append").saveAsTable(f"{destination_catalog}.{log_database}.{log_sync_tables}")

  except Exception as e:
    error = str(e).replace("'", "\\'")
    print(f"ERROR: Unable to update the log table {destination_catalog}.{log_database}.{log_sync_tables}: {error}")

# COMMAND ----------

# Sync tables to UC

execution_time = datetime.datetime.now()

if dry_run:
    # Execute an assessment in the entire inventory
    with ThreadPoolExecutor(max_workers = 8) as executor:
        executor.map(sync_tables, source_catalog_tables_valid.collect(), itertools.repeat(batch_id), itertools.repeat(dry_run), itertools.repeat(execution_time))
else:
    ############### Migrate External Tables ###############
    external_tables = source_catalog_tables_valid.where(col("table_type") == "EXTERNAL")
    with ThreadPoolExecutor(max_workers = 8) as executor:
        executor.map(sync_tables, external_tables.collect(), itertools.repeat(batch_id), itertools.repeat(dry_run), itertools.repeat(execution_time))

    ############### Migrate Managed Tables ###############
    managed_tables = source_catalog_tables_valid.where(col("table_type") == "MANAGED")
    with ThreadPoolExecutor(max_workers = 8) as executor:
        executor.map(migrate_managed_tables, managed_tables.collect(), itertools.repeat(batch_id), itertools.repeat(dry_run), itertools.repeat(execution_time))

    ############### Migrate Views ###############
    views = source_catalog_tables_valid.where(col("table_type") == "VIEW").select("db_name", "table_name")
    view_list = [ f"{row.db_name}.{row.table_name}" for row in views.collect() ]

    # Create list of views and dependencies
    list_view_dependencies = {}
    for view_obj in view_list:
        create_list_view_dependencies(view_obj, list_view_dependencies)

    # Sort the list in a way to attemp migrating the dependencies first
    resolved = []
    seen = []
    for key, value in list_view_dependencies.items():
        dep_resolve(key, value, list_view_dependencies, resolved, seen)

    # Creating a list with unique values
    resolved_unique = []
    for item in resolved:
        if item not in resolved_unique:
            resolved_unique.append(item)

    # Creating the final list of views to migrate
    final_view_list = [x for x in resolved_unique if x in view_list]

    # Migrating views
    for view_name in final_view_list:
        migrate_views(view_name, batch_id, dry_run, execution_time)

# COMMAND ----------

# Optimizing log table
spark.sql(f"OPTIMIZE {log_table_name}")

# COMMAND ----------

display(spark.sql(f"select * from {log_table_name} where batch_id = {batch_id}"))

# COMMAND ----------

# MAGIC %md
# MAGIC # PART 5: Change ownership of migrated tables

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
