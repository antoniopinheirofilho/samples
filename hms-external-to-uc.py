# Databricks notebook source
# MAGIC %md
# MAGIC # External tables to UC managed tables
# MAGIC
# MAGIC This notebook will migrate all external tables from a Hive metastore to a UC catalog.
# MAGIC
# MAGIC **Important:**
# MAGIC - This notebook needs to run in a cluster with connection to the object storage where the external tables data objects are located
# MAGIC - All external locations related to the external tables must have been criated prior to running this notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

from pyspark.sql.functions import col
from concurrent.futures import ThreadPoolExecutor
import os

# Source and target information
source_catalog = "hive_metastore"
destination_catalog = "prodtestcatalog"

# Log Database
log_database = "uc_migration_log_db"
log_sync_tables = "uc_migration_log_tables_tb"

# Parallelism when syncing the tables
default_parallelism = os.cpu_count()

# COMMAND ----------

# List of Databases to have their external tables migrated

#external_tables_dbs = ["test_uc_migration_1", "test_uc_migration_2", "test_uc_migration_3"]
external_tables_dbs = ["test_uc_migration_1", "test_uc_migration_2", "test_uc_migration_3", "test_uc_migration_1_non_delta", "test_uc_migration_2_non_delta"]

# COMMAND ----------

# MAGIC %md
# MAGIC # PART I: Get all tables from Hive Metastore

# COMMAND ----------

def get_value(lst, idx, idy, default):
    try:
        return lst[idx][idy]
    except IndexError:
        return default

# COMMAND ----------

descriptions = []

# Loop through each database
for db in external_tables_dbs:

    # Get all tables from the current database
    print(f"Start processing Database {db}")
    tables = spark.sql("show tables in {}".format(db)).select("tableName").collect()

    # Loop through each table and run the describe command
    for table in tables:
        table_name = table.tableName
        try:
            desc = spark.sql(f"DESCRIBE FORMATTED {db}.{table_name}").filter("col_name = 'Location' OR col_name='Database' OR col_name='Table' OR col_name='Type' OR col_name='Provider'")
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

            descriptions.append((database_name, table_name, table_type, table_location, table_provider, ""))
        except Exception as ex:
          descriptions.append((db, table_name, None, None, None, str(ex)))
            
# Create DataFrame from the results
source_catalog_tables = spark.createDataFrame(descriptions, ['database_name', 'table_name', 'table_type', 'table_location', 'table_provider', 'error'])
display(source_catalog_tables)

# COMMAND ----------

# MAGIC %md
# MAGIC # PART II: Select the external tables to migrate
# MAGIC
# MAGIC - Only external tables are supported by the UC Sync command

# COMMAND ----------

table_inventory = source_catalog_tables.where(col("table_location").startswith("abfss"))
display(table_inventory)

# COMMAND ----------

# MAGIC %md
# MAGIC # PART III: Remaining tables will need to have their data migrated phisically using delta clone

# COMMAND ----------

remaining_tables = source_catalog_tables.exceptAll(table_inventory)
display(remaining_tables)

# COMMAND ----------

# MAGIC %md
# MAGIC # PART IV: Upgrade tables to UC

# COMMAND ----------

# Creating Databases if they don't exist

for db in table_inventory.select("database_name").distinct().collect():
  try:
    db_name = db["database_name"]
    print(f"Creating Database/Schema {db_name}")
    spark.sql(f"CREATE DATABASE IF NOT EXISTS`{destination_catalog}`.{db_name}")
  except Exception as ex:
    print(f"Unable to create database {db_name}: {str(ex)}")

# COMMAND ----------

def sync_tables(inventory_obj):

  db_name = inventory_obj["database_name"]
  tb_name = inventory_obj["table_name"]
  tb_location = inventory_obj["table_location"]
  tb_provider = inventory_obj["table_provider"]
  full_name_source = f"{source_catalog}.{db_name}.{tb_name}"
  full_name_target = f"{destination_catalog}.{db_name}.{tb_name}"
  log = []

  try:

    # Drop table in UC if it already exists
    print(f"Dropping table {full_name_target}")
    spark.sql(f"DROP TABLE IF EXISTS {full_name_target}")

    # Create external table in Unity Catalog
    print(f"Creating table {full_name_target} from location '{tb_location}' in {tb_provider} format")

    if tb_provider == "delta":
      df_sync_result = spark.sql(f"create table if not exists {full_name_target} location '{tb_location}'")
    elif tb_provider == "parquet":
      df_sync_result = spark.sql(f"create table if not exists {full_name_target} using parquet location '{tb_location}'")
    else:
      raise Exception(f"Table format {tb_provider} is not supported")

    # Generating log
    log.append((full_name_source, full_name_target, tb_location, tb_provider, "OK", ""))

  except Exception as e:
    error = str(e).replace("'", "\\'")
    print(f"Unable to create table {full_name_source} to {full_name_target} of type {tb_provider}: {error}")
    # Generating log
    log.append((full_name_source, full_name_target, tb_location, tb_provider, "ERROR", error))

  source_catalog_tables = spark.createDataFrame(log, ['source_table', 'target_table', 'table_location', 'table_provider', 'status', 'error'])

  # Save result in a log table
  source_catalog_tables.write.format("delta").mode("append").saveAsTable(f"{destination_catalog}.{log_database}.{log_sync_tables}")

# COMMAND ----------

# Sync tables to UC

# Setting up log table
log_table_name = f"{destination_catalog}.{log_database}.{log_sync_tables}"
spark.sql(f"CREATE DATABASE IF NOT EXISTS`{destination_catalog}`.{log_database}")  
spark.sql(f"CREATE OR REPLACE TABLE {log_table_name} (source_table STRING, target_table STRING, table_location STRING, table_provider STRING, status STRING, error STRING)")

# Executing Sync command in a thread pull
with ThreadPoolExecutor(max_workers = default_parallelism) as executor:
    executor.map(sync_tables, table_inventory.collect())

# Optimizing log table
spark.sql(f"OPTIMIZE {log_table_name}")

# COMMAND ----------

display(spark.sql(f"select * from {log_table_name}"))
