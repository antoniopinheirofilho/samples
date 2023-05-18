# Databricks notebook source
from pyspark.sql.functions import *
from databricks import *
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup
import uuid

# COMMAND ----------

raw_data = spark.read.load("/databricks-datasets/wine-quality/winequality-red.csv",format="csv",sep=";",inferSchema="true",header="true" )

def addIdColumn(dataframe, id_column_name):
    """Add id column to dataframe"""
    columns = dataframe.columns
    new_df = dataframe.withColumn(id_column_name, monotonically_increasing_id())
    return new_df[[id_column_name] + columns]

def renameColumns(df):
    """Rename columns to be compatible with Feature Store"""
    renamed_df = df
    for column in df.columns:
        renamed_df = renamed_df.withColumnRenamed(column, column.replace(' ', '_'))
    return renamed_df

# Run functions
renamed_df = renameColumns(raw_data)
df = addIdColumn(renamed_df, 'wine_id')

# Drop target column ('quality') as it is not included in the feature table
features_df = df.drop('quality')
display(features_df)

# COMMAND ----------

table_name = f"test_us_fs_integration.test_us_fs_integration_db.wine_db_" + str(uuid.uuid4())[:6]
print(table_name)

# COMMAND ----------

fs = feature_store.FeatureStoreClient()
fs.create_table(
    name=table_name,
    primary_keys=["wine_id"],
    df=features_df,
    schema=features_df.schema,
    description="wine features"
)

# COMMAND ----------

df_fs = spark.sql(f"select * from {table_name}")
df_fs = df_fs.where(col("free_sulfur_dioxide") > 10)
fs.create_table(
    name=table_name + "_v2",
    primary_keys=["wine_id"],
    df=df_fs,
    schema=df_fs.schema
)

# COMMAND ----------

df_fs_2 = spark.read.table(table_name)
df_fs_2 = df_fs_2.where(col("free_sulfur_dioxide") > 10)
fs.create_table(
    name=table_name + "_v3",
    primary_keys=["wine_id"],
    df=df_fs_2,
    schema=df_fs_2.schema
)

# COMMAND ----------

df_fs_3 = fs.read_table(name=table_name)
df_fs_3 = df_fs_3.where(col("free_sulfur_dioxide") > 10)
fs.create_table(
    name=table_name + "_v3",
    primary_keys=["wine_id"],
    df=df_fs_3,
    schema=df_fs_3.schema
)
