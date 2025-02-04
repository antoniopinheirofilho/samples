# Databricks notebook source
# MAGIC %md
# MAGIC # External HMS tables to UC
# MAGIC
# MAGIC This notebook will migrate tables from a Hive metastore to a UC catalog.
# MAGIC
# MAGIC **Important:**
# MAGIC - The cluster used by this notebook must support Unity Catalog: https://docs.databricks.com/en/compute/configure.html#access-modes
# MAGIC - The cluster used by this notebook must be configured to access the underlying HMS ADLS locations. This connectivity can be configured either on the cluster level or on the code level: https://docs.databricks.com/en/connect/storage/azure-storage.html#set-spark-properties-to-configure-azure-credentials-to-access-azure-storage
# MAGIC - All external locations related to the tables in scope must be registered in Unity Catalog as External Locations. Also, the user running this script and creating tables must have CREATE TABLE permission on the External Locations
# MAGIC - The Spark Property "spark.databricks.sync.command.enableManagedTable" must be set to True in the Cluster Level to allow the migration of MANAGED Tables with MANAGED Storage Locations
# MAGIC - By default, the SYNC statement adds a property to the existing delta table. To prevent SYNC from adding this property, set spark.databricks.sync.command.disableSourceTableWrites to True
# MAGIC
# MAGIC **Input Parameters:**
# MAGIC - inventory_query: SQL statement that generates the inventory of tables to be migrated. The dataset returned by this SQL statement must contain the column source_catalog, source_database, source_table, target_catalog, target_database, target_table, table_type (MANAGED, EXTERNAL or VIEW), table_provider (DELTA, PARQUET, ETC), table_locationFor example:
# MAGIC
# MAGIC `select source_catalog, source_database, source_table, target_catalog, target_database, target_table, table_type, table_provider, table_location 
# MAGIC from table_inventory`
# MAGIC
# MAGIC - is_dry_run: Indicates whether the job will execute a dry run or actually migrate the tables. This functionality will simulate migrations for only EXTERNAL tables, since MANAGED tables and views don't support dry runs
# MAGIC - table_owner_uc: The principal who should own the tables in UC after migration. The script will change the table ownership to this user at the end of the process
# MAGIC - log_catalog: Catalog where this script will store the migration logs
# MAGIC - log_database: Database where this script will store the migration logs
# MAGIC - log_migration_table: Table where this script will store the migration logs

# COMMAND ----------

# MAGIC %md
# MAGIC # PART 0: Setup connectivity with ADLS
# MAGIC
# MAGIC The UC needs to be able to access ADLS via legacy HMS, which is configured via spark configs either on the cluster level or on the code level

# COMMAND ----------

# Add the ADLS connection configuration here is applicable

# COMMAND ----------

# MAGIC %md
# MAGIC # PART 1: Configuration

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC pip install sqlglot

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql.functions import *
from concurrent.futures import ThreadPoolExecutor
import itertools 
import os
import datetime
import re

# COMMAND ----------

import time

dbutils.widgets.removeAll()

time.sleep(3)

# COMMAND ----------

dbutils.widgets.text("inventory_query", """
                     
select
    source_catalog,
    source_database,
    source_table,
    target_catalog,
    target_database,
    target_table,
    table_type,
    table_provider,
    table_location
from hive_metastore.default.table_inventory
                     
                     """)
dbutils.widgets.text("is_dry_run", "False")
dbutils.widgets.text("table_owner_uc", "")
dbutils.widgets.text("log_catalog", "test_uc_migration")
dbutils.widgets.text("log_database", "uc_migration_log_db")
dbutils.widgets.text("log_migration_table", "uc_table_migration_log")

# COMMAND ----------

inventory_query = dbutils.widgets.get("inventory_query")
is_dry_run = dbutils.widgets.get("is_dry_run")
table_owner = dbutils.widgets.get("table_owner_uc")
log_catalog = dbutils.widgets.get("log_catalog")
log_database = dbutils.widgets.get("log_database")
log_migration_table = dbutils.widgets.get("log_migration_table")

# It will create as many threads as the number of cores available in the driver node
default_parallelism = os.cpu_count()

# COMMAND ----------

# MAGIC %md
# MAGIC # PART 2: Table inventory and mapping

# COMMAND ----------

df_inventory_mapping = spark.sql(inventory_query)
display(df_inventory_mapping)

# COMMAND ----------

# MAGIC %md
# MAGIC # PART 3: Parsing and safety checks

# COMMAND ----------

# Safety check
if df_inventory_mapping.where((lower(trim(col("target_catalog"))) == "hive_metastore") | (lower(trim(col("target_catalog"))) == lower(trim(col("source_catalog"))))).count() > 0:
    raise Exception("ERROR: Invalid destionation catalog.") 

# Set table Owner in UC
if not table_owner.strip():
    raise Exception("ERROR: UC table owner must be specified.") 

# Is this a dry run?
dry_run = True
if is_dry_run.upper() == "FALSE":
    dry_run = False

# COMMAND ----------

# MAGIC %md
# MAGIC # PART 4: Create Migration Log Tables

# COMMAND ----------

# Setting up log table
log_table_name = f"{log_catalog}.{log_database}.{log_migration_table}"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {log_catalog}.{log_database}")  
spark.sql(f"CREATE TABLE IF NOT EXISTS {log_table_name} (source_table STRING, target_table STRING, table_location STRING, table_type STRING, table_provider STRING, command STRING, status STRING, description STRING, dry_run BOOLEAN, batch_id LONG, execution_time TIMESTAMP)")

# Generating batch id
batch_id = spark.sql(f"select max(batch_id) + 1 from {log_table_name}").collect()[0][0]

if not batch_id:
    batch_id = 1

# COMMAND ----------

# MAGIC %md
# MAGIC # PART 5: Helpers for external table migration

# COMMAND ----------

def migrate_external_tables(inventory_obj, batch_id, dry_run, execution_time):

  source_catalog = inventory_obj["source_catalog"]
  source_database = inventory_obj["source_database"]
  source_table = inventory_obj["source_table"]
  target_catalog = inventory_obj["target_catalog"]
  target_database = inventory_obj["target_database"]
  target_table = inventory_obj["target_table"] 
  table_location = inventory_obj["table_location"]
  table_provider = inventory_obj["table_provider"]
  table_type = inventory_obj["table_type"]
  full_name_source = f"{source_catalog}.{source_database}.{source_table}"
  full_name_target = f"{target_catalog}.{target_database}.{target_table}"
  command = ""
  status_code = ""
  descriptions = ""
  log = []

  try:

    # Create external table in Unity Catalog
    if table_provider == "hive":
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
    log.append((full_name_source, full_name_target, table_location, table_provider, table_type, command, status_code, descriptions, dry_run, batch_id, execution_time))

  except Exception as e:
    error = str(e).replace("'", "\\'")
    print(f"Unable to create table {full_name_source} to {full_name_target} of type {table_provider}: {error}")
    # Generating log
    log.append((full_name_source, full_name_target, table_location, table_provider, table_type, command, "ERROR", error, dry_run, batch_id, execution_time))

  # Save result in a log table
  try:

    source_catalog_tables = spark.createDataFrame(log, ['source_table', 'target_table', 'table_location', 'table_provider', 'table_type', 'command', 'status', 'description', 'dry_run', 'batch_id', 'execution_time'])

    source_catalog_tables.write.format("delta").mode("append").saveAsTable(f"{log_catalog}.{log_database}.{log_migration_table}")

  except Exception as e:
    error = str(e).replace("'", "\\'")
    print(f"ERROR: Unable to update the log table {log_catalog}.{log_database}.{log_migration_table}: {error}")

# COMMAND ----------

# MAGIC %md
# MAGIC # PART 6: Helpers for managed table migration

# COMMAND ----------

def migrate_managed_tables(inventory_obj, batch_id, dry_run, execution_time):

  catalog_name_source = inventory_obj["source_catalog"]
  catalog_name_target = inventory_obj["target_catalog"]
  db_name_source = inventory_obj["source_database"]
  db_name_target = inventory_obj["target_database"]
  tb_name_source = inventory_obj["source_table"]
  tb_name_target = inventory_obj["target_table"]
  tb_location = inventory_obj["table_location"]
  tb_provider = inventory_obj["table_provider"]
  tb_type = inventory_obj["table_type"]
  full_name_source = f"{catalog_name_source}.{db_name_source}.{tb_name_source}"
  full_name_target = f"{catalog_name_target}.{db_name_target}.{tb_name_target}"
  command = ""
  status_code = ""
  descriptions = ""
  log = []

  try:

    # Create external table in Unity Catalog
    if tb_provider.lower() == "delta":
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

    source_catalog_tables.write.format("delta").mode("append").saveAsTable(f"{log_catalog}.{log_database}.{log_migration_table}")

  except Exception as e:
    error = str(e).replace("'", "\\'")
    print(f"ERROR: Unable to update the log table {log_catalog}.{log_database}.{log_migration_table}: {error}")

# COMMAND ----------

# MAGIC %md
# MAGIC # PART 7: Helpers for view migration

# COMMAND ----------

import re
from sqlglot import parse_one, exp

def create_list_view_dependencies(view_name, list_view_dependencies):
  
  full_name_source = f"hive_metastore.{view_name}"

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

def migrate_views(view_name, target_catalog, batch_id, dry_run, execution_time):

  full_name_source = f"hive_metastore.{view_name}"
  full_name_target = f"{target_catalog}.{view_name}"
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

    new_create_table = old_create_table.replace("hive_metastore.", "")
    for full_tb_name in list_dependencies:
      full_uc_table_name = f"{target_catalog}.{full_tb_name}"
      new_create_table = re.sub(r'\b{}\b'.format(re.escape(full_tb_name)), full_uc_table_name, new_create_table)

    new_create_table = new_create_table.replace("CREATE VIEW", "CREATE OR REPLACE VIEW")

    # Safety check
    if new_create_table.startswith(f"CREATE OR REPLACE VIEW {target_catalog}."):
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

    source_catalog_tables.write.format("delta").mode("append").saveAsTable(f"{log_catalog}.{log_database}.{log_migration_table}")

  except Exception as e:
    error = str(e).replace("'", "\\'")
    print(f"ERROR: Unable to update the log table {log_catalog}.{log_database}.{log_migration_table}: {error}")

# COMMAND ----------

# MAGIC %md
# MAGIC # PART 8: Run migration

# COMMAND ----------

# DBTITLE 1,External Table Inventory
external_tables = df_inventory_mapping.where((col("table_type") == "EXTERNAL") |  (col("table_location").startswith("dbfs:/mnt")) | (col("table_location").startswith("abfss://")))
display(external_tables)

# COMMAND ----------

# DBTITLE 1,Managed Table Inventory
managed_tables = df_inventory_mapping.where((col("table_type") == "MANAGED") & (~col("table_location").startswith("dbfs:/mnt")) & (~col("table_location").startswith("abfss://")) )
display(managed_tables)

# COMMAND ----------

# DBTITLE 1,Views Inventory
views_full = df_inventory_mapping.where(col("table_type") == "VIEW")
views = views_full.select("source_catalog", "source_database", "source_table")
display(views)

# COMMAND ----------

# Sync tables to UC

execution_time = datetime.datetime.now()

if dry_run:
    # Execute an assessment in the entire inventory
    with ThreadPoolExecutor(max_workers = default_parallelism) as executor:
        executor.map(migrate_external_tables, external_tables.collect(), itertools.repeat(batch_id), itertools.repeat(dry_run), itertools.repeat(execution_time))
else:
    ############### Migrate External Tables ###############
    with ThreadPoolExecutor(max_workers = default_parallelism) as executor:
        executor.map(migrate_external_tables, external_tables.collect(), itertools.repeat(batch_id), itertools.repeat(dry_run), itertools.repeat(execution_time))

    ############### Migrate Managed Tables ###############
    with ThreadPoolExecutor(max_workers = default_parallelism) as executor:
        executor.map(migrate_managed_tables, managed_tables.collect(), itertools.repeat(batch_id), itertools.repeat(dry_run), itertools.repeat(execution_time))

    ############### Migrate Views ###############
    views_dict = [row.asDict() for row in views_full.collect()]
    view_list = [ f"{row.source_database}.{row.source_table}" for row in views.collect() ]

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

        v = view_name.split(".")
        db = v[0]
        tb = v[1]
        target_catalog = next((item for item in views_dict if item['target_database'] == db and item['target_table'] == tb), None)["target_catalog"]

        migrate_views(view_name, target_catalog, batch_id, dry_run, execution_time)

# COMMAND ----------

# Optimizing log table
spark.sql(f"OPTIMIZE {log_table_name}")

# COMMAND ----------

display(spark.sql(f"select * from {log_table_name} where batch_id = {batch_id}"))

# COMMAND ----------

# MAGIC %md
# MAGIC # PART 9: Change ownership of migrated tables

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
