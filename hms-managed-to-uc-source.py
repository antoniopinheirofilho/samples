# Databricks notebook source
# MAGIC %md
# MAGIC # Managed tables to UC external tables - Migrate data with Delta DEEP CLONE
# MAGIC
# MAGIC This notebook will migrate all managed tables from a Hive metastore to a UC catalog.
# MAGIC
# MAGIC **Important:**
# MAGIC - This script needs to run in the source environment
# MAGIC - The source environment cluster needs to have access to the external ADLS location where the managed data will be migrated/cloned to
# MAGIC - This notebook requires a list of managed tables to be migrated along with the external ADLS location where it will be migrated to
# MAGIC - This notebook will create clones of the managed tables in HMS for it to be able to migrate data from the managed location to an external location

# COMMAND ----------

from pyspark.sql.functions import col
from concurrent.futures import ThreadPoolExecutor
import os

# Log Database
log_database = "uc_migration_log_db"
log_table = "uc_migration_log_tables_tb"

# Database in HMS where the cloned tables will be created at
cloned_tables_db = "cloned_tables_db"

# Prefix of the cloned tables
cloned_tables_prefix = "uc_migration_clone"

# Parallelism when syncing the tables
default_parallelism = os.cpu_count()

# COMMAND ----------

# MAGIC %md
# MAGIC # Step I: Define the list of managed tables to migrated
# MAGIC
# MAGIC - This is a dataset containing the source database name, source tables name and external location where we will migrate the data objects to
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
# MAGIC
# MAGIC # Step II: Migrate managed table using Delta DEEP CLONE
# MAGIC
# MAGIC - This function creates a deep clone of a managed table
# MAGIC - The cloned table is unmanaged. Therefore, the data objects will reside in an external location
# MAGIC - This process can be executed idempotently - DEEP Clone will only transfer the latest managed tables changes since the last execution. Therefore, the first run might take time, but the subsequent ones should be a lot faster
# MAGIC - Once this is done, we can create external tables in the UC workspace pointing to these external locations

# COMMAND ----------

import re
import json

def exec_migrate_table(managed_table_obj):

    db_name = managed_table_obj["db_name"]
    table_name = managed_table_obj["table_name"]
    target_location = managed_table_obj["target_location"]

    # Cloned table which will be created in HMS
    cloned_table_name = f"{cloned_tables_prefix}_{db_name}_{table_name}"

    print("Creating clonned table {}.{} from table {}.{} in location {}".format(cloned_tables_db, cloned_table_name, db_name, table_name, target_location))
    
    log = []

    try:

      spark.sql("CREATE OR REPLACE TABLE {0}.{1} DEEP CLONE {2}.{3} LOCATION '{4}'".format(cloned_tables_db, cloned_table_name, db_name, table_name, target_location))
      log.append((db_name, table_name, cloned_tables_db, cloned_table_name, target_location, "OK", ""))

    except Exception as e:
      error = str(e).replace("'", "\\'")
      log.append((db_name, table_name, cloned_tables_db, cloned_table_name, target_location, "ERROR", error))

    source_internal_tables = spark.createDataFrame(log, ['source_db_name', 'source_table_name', 'cloned_db_name', 'cloned_table_name', 'external_location', 'status', 'error'])

    # Save result in a log table
    source_internal_tables.write.format("delta").mode("append").saveAsTable(f"{log_database}.{log_table}")

# COMMAND ----------

# Sync tables to UC

# Setting up log table
log_table_name = f"{log_database}.{log_table}"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {log_database}")  
spark.sql(f"CREATE OR REPLACE TABLE {log_table_name} (source_db_name STRING, source_table_name STRING, cloned_db_name STRING, cloned_table_name STRING, external_location STRING, status STRING, error STRING)")

# Creating cloned tables database
spark.sql(f"CREATE DATABASE IF NOT EXISTS {cloned_tables_db}")

# Executing Sync command in a thread pull
with ThreadPoolExecutor(max_workers = default_parallelism) as executor:
    executor.map(exec_migrate_table, managed_tables_list)

# Optimizing log table
spark.sql(f"OPTIMIZE {log_table_name}")

# COMMAND ----------

display(spark.sql(f"select * from {log_table_name}"))
