# Databricks notebook source
# MAGIC %md
# MAGIC # Managed tables to UC external tables - Create metadata pointing to external location
# MAGIC
# MAGIC This notebook will create UC table metadata pointing to the external location where the managed tables data were migrated to.
# MAGIC
# MAGIC **Important:**
# MAGIC - This script needs to run in the target environment
# MAGIC - The source environment cluster needs to have access to the external ADLS location where the data objects were migrated to
# MAGIC - This notebook requires a list of managed tables from source to be created as external in the target environment
# MAGIC - This should run only after executing the notebook hms-managed-to-uc-source

# COMMAND ----------

from pyspark.sql.functions import col
from concurrent.futures import ThreadPoolExecutor
import os

# target information
destination_catalog = "prodtestcatalog"

# Log Database
log_database = "uc_migration_log_db"
log_table = "uc_migration_log_tables_tb"

# Parallelism when syncing the tables
default_parallelism = os.cpu_count()

# COMMAND ----------

# MAGIC %md
# MAGIC # Step I: List of managed tables from source
# MAGIC
# MAGIC - Same dataset used in the notebook hms-managed-to-uc-source
# MAGIC - This can either be hard-coded as below, or retrieved from a file

# COMMAND ----------

# List of Managed Tables in scope

managed_tables_list = [{"db_name": "test_uc_migration_1", "table_name": "internal_table_1", "target_location": ""},
 {"db_name": "test_uc_migration_1", "table_name": "internal_table_2", "target_location": ""},
 {"db_name": "test_uc_migration_2", "table_name": "internal_table_1", "target_location": ""},
 {"db_name": "test_uc_migration_2", "table_name": "internal_table_2", "target_location": ""},
 {"db_name": "test_uc_migration_3", "table_name": "internal_table_1", "target_location": ""},
 {"db_name": "test_uc_migration_3", "table_name": "internal_table_2", "target_location": ""}]

# COMMAND ----------

# MAGIC %md
# MAGIC # Step II: Create databases in UC

# COMMAND ----------

# Creating Databases if they don't exist

for db in managed_tables_list:
  try:
    db_name = db["db_name"]
    print(f"Creating Database/Schema {destination_catalog}.{db_name}")
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {destination_catalog}.{db_name}")
  except Exception as ex:
    print(f"Unable to create database {destination_catalog}.{db_name}: {str(ex)}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Step III: Creating UC tables metadata
# MAGIC
# MAGIC - This will create the UC tables representing the managed tables in the source environment
# MAGIC - The external tables created here will point to the external data objects migrated by the notebook hms-managed-to-uc-source

# COMMAND ----------

def create_uc_tables(inventory_obj):

  db_name = inventory_obj["db_name"]
  tb_name = inventory_obj["table_name"]
  tb_location = inventory_obj["target_location"]
  full_name_target = f"{destination_catalog}.{db_name}.{tb_name}"
  log = []

  try:

    # Drop table in UC if it already exists
    print(f"Dropping table {full_name_target}")
    spark.sql(f"DROP TABLE IF EXISTS {full_name_target}")

    # Create external table in Unity Catalog
    print(f"Creating table {full_name_target} from location '{tb_location}'")
    df_sync_result = spark.sql(f"create table if not exists {full_name_target} location '{tb_location}'")

    # Generating log
    log.append((db_name, tb_name, full_name_target, tb_location, "OK", ""))

  except Exception as e:
    error = str(e).replace("'", "\\'")
    print(f"Unable to sync table {full_name_source} to {full_name_target}: {error}")
    # Generating log
    log.append((db_name, tb_name, full_name_target, tb_location, "ERROR", error))

  source_catalog_tables = spark.createDataFrame(log, ['source_database', 'source_table', 'target_table', 'target_table_location', 'status', 'error'])

  # Save result in a log table
  source_catalog_tables.write.format("delta").mode("append").saveAsTable(f"{destination_catalog}.{log_database}.{log_table}")

# COMMAND ----------

# Sync tables to UC

# Setting up log table
log_table_name = f"{destination_catalog}.{log_database}.{log_table}"
spark.sql(f"CREATE DATABASE IF NOT EXISTS`{destination_catalog}`.{log_database}")  
spark.sql(f"CREATE OR REPLACE TABLE {log_table_name} (source_database STRING, source_table STRING, target_table STRING, target_table_location STRING, status STRING, error STRING)")

# Executing Sync command in a thread pull
with ThreadPoolExecutor(max_workers = default_parallelism) as executor:
    executor.map(create_uc_tables, managed_tables_list)

# Optimizing log table
spark.sql(f"OPTIMIZE {log_table_name}")

# COMMAND ----------

display(spark.sql(f"select * from {log_table_name}"))

# COMMAND ----------


