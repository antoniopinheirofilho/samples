# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install -U --quiet databricks-sdk mlflow textstat tiktoken evaluate

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Parameters

# COMMAND ----------

endpoint_host = "https://adb-2332510266816567.7.azuredatabricks.net"
endpoint_name = "llm_prod_endpoint"
inference_table_name = "llmops_prod.model_tracking.rag_app_realtime_payload"
inference_processed_table = "llmops_prod.model_tracking.rag_app_realtime_payload_processed"
endpoint_token = dbutils.secrets.get(scope="creds", key="pat")
working_dir = "/dbfs/tmp/llmops_prod/rag_app"

# COMMAND ----------

# MAGIC %md
# MAGIC # Populate inference table

# COMMAND ----------

import requests
import json

def get_answer(question):

    data = {
        "input": question
    }

    headers = {"Context-Type": "text/json", "Authorization": f"Bearer {endpoint_token}"}

    response = requests.post(
        url=f"{endpoint_host}/serving-endpoints/{endpoint_name}/invocations", json=data, headers=headers
    )

    return response.text

# COMMAND ----------

get_answer("How to start a cluster?")

# COMMAND ----------

get_answer("How to bake a cake?")

# COMMAND ----------

get_answer("How to create a notebook?")

# COMMAND ----------

get_answer("How to rob a bank?")

# COMMAND ----------

get_answer("How to see a job run?")

# COMMAND ----------

display(spark.table(inference_table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC # Unpack the table

# COMMAND ----------

import time
import re
import io
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.functions import col, pandas_udf, transform, size, element_at


def unpack_requests(requests_raw: DataFrame, 
                    input_request_json_path: str, 
                    input_json_path_type: str, 
                    output_request_json_path: str, 
                    output_json_path_type: str,
                    keep_last_question_only: False) -> DataFrame:
    # Rename the date column and convert the timestamp milliseconds to TimestampType for downstream processing.
    requests_timestamped = (requests_raw
        .withColumn("timestamp", (col("timestamp_ms") / 1000))
        .drop("timestamp_ms"))

    # Convert the model name and version columns into a model identifier column.
    requests_identified = requests_timestamped.withColumn(
        "model_id",
        F.concat(
            col("request_metadata").getItem("model_name"),
            F.lit("_"),
            col("request_metadata").getItem("model_version")
        )
    )

    # Filter out the non-successful requests.
    requests_success = requests_identified.filter(col("status_code") == 200)

    # Unpack JSON.
    requests_unpacked = (requests_success
        .withColumn("request", F.from_json(F.expr(f"request:{input_request_json_path}"), input_json_path_type))
        .withColumn("response", F.from_json(F.expr(f"response:{output_request_json_path}"), output_json_path_type)))
    
    if keep_last_question_only:
        requests_unpacked = requests_unpacked.withColumn("request", F.array(F.element_at(F.col("request"), -1)))

    # Explode batched requests into individual rows.
    requests_exploded = (requests_unpacked
        .withColumn("__db_request_response", F.explode(F.arrays_zip(col("request").alias("input"), col("response").alias("output"))))
        .selectExpr("* except(__db_request_response, request, response, request_metadata)", "__db_request_response.*")
        )

    return requests_exploded

# COMMAND ----------

# The format of the input payloads, following the TF "inputs" serving format with a "query" field.
# Single query input format: {"inputs": [{"query": "User question?"}]}
INPUT_REQUEST_JSON_PATH = "inputs[*].query"

# Matches the schema returned by the JSON selector (inputs[*].query is an array of string)
INPUT_JSON_PATH_TYPE = "array<string>"
KEEP_LAST_QUESTION_ONLY = False

# Answer format: {"predictions": ["answer"]}
OUTPUT_REQUEST_JSON_PATH = "predictions"

# Matches the schema returned by the JSON selector (predictions is an array of string)
OUPUT_JSON_PATH_TYPE = "array<string>"

# COMMAND ----------

payloads_sample_df = spark.table(inference_table_name).where('status_code == 200').limit(10)
display(payloads_sample_df)

# COMMAND ----------

# Unpack using provided helper function
payloads_unpacked_sample_df = unpack_requests(
    payloads_sample_df,
    INPUT_REQUEST_JSON_PATH,
    INPUT_JSON_PATH_TYPE,
    OUTPUT_REQUEST_JSON_PATH,
    OUPUT_JSON_PATH_TYPE,
    KEEP_LAST_QUESTION_ONLY
)

display(payloads_unpacked_sample_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Compute Metrics

# COMMAND ----------

import tiktoken, textstat, evaluate
import pandas as pd
from pyspark.sql.functions import pandas_udf


@pandas_udf("int")
def compute_num_tokens(texts: pd.Series) -> pd.Series:
  encoding = tiktoken.get_encoding("cl100k_base")
  return pd.Series(map(len, encoding.encode_batch(texts)))

@pandas_udf("double")
def flesch_kincaid_grade(texts: pd.Series) -> pd.Series:
  return pd.Series([textstat.flesch_kincaid_grade(text) for text in texts])
 
@pandas_udf("double")
def automated_readability_index(texts: pd.Series) -> pd.Series:
  return pd.Series([textstat.automated_readability_index(text) for text in texts])

@pandas_udf("double")
def compute_toxicity(texts: pd.Series) -> pd.Series:
  # Omit entries with null input from evaluation
  toxicity = evaluate.load("toxicity", module_type="measurement", cache_dir="/tmp/hf_cache/")
  return pd.Series(toxicity.compute(predictions=texts.fillna(""))["toxicity"]).where(texts.notna(), None)

@pandas_udf("double")
def compute_perplexity(texts: pd.Series) -> pd.Series:
  # Omit entries with null input from evaluation
  perplexity = evaluate.load("perplexity", module_type="measurement", cache_dir="/tmp/hf_cache/")
  return pd.Series(perplexity.compute(data=texts.fillna(""), model_id="gpt2")["perplexities"]).where(texts.notna(), None)

# COMMAND ----------

from pyspark.sql import DataFrame
from pyspark.sql.functions import col


def compute_metrics(requests_df: DataFrame, column_to_measure = ["input", "output"]) -> DataFrame:
  for column_name in column_to_measure:
    requests_df = (
      requests_df.withColumn(f"toxicity({column_name})", compute_toxicity(col(column_name)))
                 .withColumn(f"perplexity({column_name})", compute_perplexity(col(column_name)))
                 .withColumn(f"token_count({column_name})", compute_num_tokens(col(column_name)))
                 .withColumn(f"flesch_kincaid_grade({column_name})", flesch_kincaid_grade(col(column_name)))
                 .withColumn(f"automated_readability_index({column_name})", automated_readability_index(col(column_name)))
    )
  return requests_df

# COMMAND ----------

# MAGIC %md
# MAGIC # Incrementally unpack & compute metrics from payloads and save to final `_processed` table 

# COMMAND ----------

import os

# Reset checkpoint [for demo purposes ONLY]
checkpoint_location = os.path.join(working_dir, "checkpoint")
dbutils.fs.rm(checkpoint_location, True)

# Unpack the requests as a stream.
requests_raw_df = spark.readStream.table(inference_table_name)
requests_processed_df = unpack_requests(
    requests_raw_df,
    INPUT_REQUEST_JSON_PATH,
    INPUT_JSON_PATH_TYPE,
    OUTPUT_REQUEST_JSON_PATH,
    OUPUT_JSON_PATH_TYPE,
    KEEP_LAST_QUESTION_ONLY
)

# Drop un-necessary columns for monitoring jobs
requests_processed_df = requests_processed_df.drop("date", "status_code", "sampling_fraction", "client_request_id", "databricks_request_id")

# Compute text evaluation metrics
requests_with_metrics_df = compute_metrics(requests_processed_df)

# COMMAND ----------

from delta.tables import DeltaTable

def create_processed_table_if_not_exists(table_name, requests_with_metrics):
    """
    Helper method to create processed table using schema
    """
    (
      DeltaTable.createOrReplace(spark) # to avoid dropping everytime .createIfNotExists(spark)
        .tableName(table_name)
        .addColumns(requests_with_metrics.schema)
        .property("delta.enableChangeDataFeed", "true")
        .property("delta.columnMapping.mode", "name")
        .execute()
    )

# COMMAND ----------

# Persist the requests stream, with a defined checkpoint path for this table

processed_table_name = f"{inference_processed_table}"
create_processed_table_if_not_exists(processed_table_name, requests_with_metrics_df)

# Append new unpacked payloads & metrics
(requests_with_metrics_df.writeStream
                      .trigger(availableNow=True)
                      .format("delta")
                      .outputMode("append")
                      .option("checkpointLocation", checkpoint_location)
                      .toTable(processed_table_name).awaitTermination())

# Display the table (with requests and text evaluation metrics) that will be monitored.
display(spark.table(processed_table_name))
