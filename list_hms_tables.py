# Databricks notebook source
from pyspark.sql.functions import *
from concurrent.futures import ThreadPoolExecutor
import itertools 
import os
import datetime
import re

# COMMAND ----------

destination_catalog = "hms_inventory_catalog"
log_database = "hms_inventory_schema"
log_scan = "hms_inventory_table"
log_scan_table_name = f"{destination_catalog}.{log_database}.{log_scan}"

spark.sql(f"CREATE DATABASE IF NOT EXISTS`{destination_catalog}`.{log_database}")  
spark.sql(f"CREATE TABLE  IF NOT EXISTS {log_scan_table_name} (db_name STRING, table_name STRING, table_type STRING, table_location STRING, table_provider STRING, error STRING)")

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Inventory of HMS tables to migrate

# COMMAND ----------

inventory = []

schemas = spark.sql(f"SHOW SCHEMAS IN hive_metastore").collect()

for s in schemas:
  
  tables = spark.sql(f"SHOW TABLES IN hive_metastore.{s.databaseName}").collect()

  for t in tables:
    inventory.append(f"hive_metastore.{s.databaseName}.{t.tableName}")

# COMMAND ----------

def scan_tables(table_full_name):
  
  descriptions = []
  database_name = ""
  table_name = ""
  table_type = ""
  table_location = ""
  table_provider = ""

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

    descriptions.append((database_name, table_name, table_type, table_location, table_provider, ""))
  except Exception as ex:
    error = str(ex).replace("'", "\\'")
    descriptions.append(("", table_full_name, "", "", "", error))

  # Save result in a log table
  try:

    scan_table_log = spark.createDataFrame(descriptions, ['db_name', 'table_name', 'table_type', 'table_location', 'table_provider', 'error'])

    scan_table_log.write.format("delta").mode("append").saveAsTable(f"{destination_catalog}.{log_database}.{log_scan}")

  except Exception as e:
    error = str(e).replace("'", "\\'")
    print(f"ERROR: Unable to update the log table {destination_catalog}.{log_database}.{log_scan}: {error}")

# COMMAND ----------

with ThreadPoolExecutor(max_workers = 4) as executor:
    executor.map(scan_tables, inventory)

# COMMAND ----------

display(spark.table(log_scan_table_name))
