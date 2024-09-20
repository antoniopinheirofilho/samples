# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install databricks-agents langchain==0.2.16 langchain-community==0.2.17 databricks-vectorsearch pydantic==1.10.9 mlflow==2.16.1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
from typing import List, Mapping, Optional

import mlflow
import mlflow.entities as mlflow_entities

from pyspark import sql
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from databricks import agents
from databricks.sdk import WorkspaceClient
from databricks.rag_eval.evaluation import traces

# COMMAND ----------

# MAGIC %md
# MAGIC # Parameters

# COMMAND ----------

uc_model_name = "llmops_prod.model_schema.basic_rag_demo_foundation_model"

# COMMAND ----------

# MAGIC %md
# MAGIC # Get the request and assessment log tables

# COMMAND ----------

w = WorkspaceClient()

active_deployments = agents.list_deployments()
active_deployment = next(
    (item for item in active_deployments if item.model_name == uc_model_name), None
)

endpoint = w.serving_endpoints.get(active_deployment.endpoint_name)

try:
    endpoint_config = endpoint.config.auto_capture_config
except AttributeError as e:
    endpoint_config = endpoint.pending_config.auto_capture_config

inference_table_name = endpoint_config.state.payload_table.name
inference_table_catalog = endpoint_config.catalog_name
inference_table_schema = endpoint_config.schema_name

# Cleanly formatted tables
assessment_log_table_name = f"{inference_table_catalog}.{inference_table_schema}.`{inference_table_name}_assessment_logs`"
request_log_table_name = f"{inference_table_catalog}.{inference_table_schema}.`{inference_table_name}_request_logs`"

print(f"Assessment logs: {assessment_log_table_name}")
print(f"Request logs: {request_log_table_name}")

assessment_log_df = spark.table(assessment_log_table_name)
request_log_df = spark.table(request_log_table_name)

# COMMAND ----------

display(assessment_log_df)

# COMMAND ----------

display(request_log_df)

# COMMAND ----------

assert assessment_log_df.count() > 0, "There is currently nothing in the assessment log table! Please interact with the review UI and wait for the assessment log to populate. If you wish to run the evaluation harness without human feedback, please use the following code: `request_log_df.toPandas()[['request', 'response', 'trace']]"

# COMMAND ----------

# MAGIC %md
# MAGIC # Remove duplicate entries from the assessment log

# COMMAND ----------

_REQUEST_ID = "request_id"
_TIMESTAMP = "timestamp"
_ROW_NUMBER = "row_number"
_SOURCE = "source"
_SOURCE_ID = "source.id"
_STEP_ID = "step_id"
_TEXT_ASSESSMENT = "text_assessment"
_RETRIEVAL_ASSESSMENT = "retrieval_assessment"


def _dedup_by_assessment_window(
    assessment_log_df: sql.DataFrame, window: Window
) -> sql.DataFrame:
    """
    Remove duplicates from the assessment logs by taking the first row from each group, defined by the window
    :param assessment_log_df: PySpark DataFrame of the assessment logs
    :param window: PySpark window to group assessments by
    :return: PySpark DataFrame of the assessment logs with duplicates removed
    """
    return (
        assessment_log_df.withColumn(_ROW_NUMBER, F.row_number().over(window))
        .filter(F.col(_ROW_NUMBER) == 1)
        .drop(_ROW_NUMBER)
    )


def _dedup_assessment_log(assessment_log_df: sql.DataFrame) -> sql.DataFrame:
    """
    Remove duplicates from the assessment logs to get the latest assessments.
    :param assessment_log_df: PySpark DataFrame of the assessment logs
    :return: PySpark DataFrame of the deduped assessment logs
    """
    # Dedup the text assessments
    text_assessment_window = Window.partitionBy(_REQUEST_ID, _SOURCE_ID).orderBy(
        F.col(_TIMESTAMP).desc()
    )
    deduped_text_assessment_df = _dedup_by_assessment_window(
        # Filter rows with null text assessments
        assessment_log_df.filter(F.col(_TEXT_ASSESSMENT).isNotNull()),
        text_assessment_window,
    )

    # Remove duplicates from the retrieval assessments
    retrieval_assessment_window = Window.partitionBy(
        _REQUEST_ID,
        _SOURCE_ID,
        f"{_RETRIEVAL_ASSESSMENT}.position",
        f"{_RETRIEVAL_ASSESSMENT}.{_STEP_ID}",
    ).orderBy(F.col(_TIMESTAMP).desc())
    deduped_retrieval_assessment_df = _dedup_by_assessment_window(
        # Filter rows with null retrieval assessments
        assessment_log_df.filter(F.col(_RETRIEVAL_ASSESSMENT).isNotNull()),
        retrieval_assessment_window,
    )

    # Collect retrieval assessments from the same request/step/source into a single list
    nested_retrieval_assessment_df = (
        deduped_retrieval_assessment_df.groupBy(_REQUEST_ID, _SOURCE_ID, _STEP_ID).agg(
            F.any_value(_TIMESTAMP).alias(_TIMESTAMP),
            F.any_value(_SOURCE).alias(_SOURCE),
            F.collect_list(_RETRIEVAL_ASSESSMENT).alias("retrieval_assessments"),
        )
        # Drop the old retrieval assessment, source id, and text assessment columns
        .drop(_RETRIEVAL_ASSESSMENT, "id", _TEXT_ASSESSMENT)
    )

    # Join the deduplicated text assessments with the nested deduplicated retrieval assessments
    deduped_assessment_log_df = deduped_text_assessment_df.alias("a").join(
        nested_retrieval_assessment_df.alias("b"),
        (F.col(f"a.{_REQUEST_ID}") == F.col(f"b.{_REQUEST_ID}"))
        & (F.col(f"a.{_SOURCE_ID}") == F.col(f"b.{_SOURCE_ID}")),
        "full_outer",
    )

    # Coalesce columns from both DataFrames in case a request does not have either assessment
    return deduped_assessment_log_df.select(
        F.coalesce(F.col(f"a.{_REQUEST_ID}"), F.col(f"b.{_REQUEST_ID}")).alias(
            _REQUEST_ID
        ),
        F.coalesce(F.col(f"a.{_STEP_ID}"), F.col(f"b.{_STEP_ID}")).alias(_STEP_ID),
        F.coalesce(F.col(f"a.{_TIMESTAMP}"), F.col(f"b.{_TIMESTAMP}")).alias(
            _TIMESTAMP
        ),
        F.coalesce(F.col(f"a.{_SOURCE}"), F.col(f"b.{_SOURCE}")).alias(_SOURCE),
        F.col(f"a.{_TEXT_ASSESSMENT}").alias(_TEXT_ASSESSMENT),
        F.col("b.retrieval_assessments").alias(_RETRIEVAL_ASSESSMENT),
    )

# COMMAND ----------

deduped_assessment_log_df = _dedup_assessment_log(assessment_log_df)
deduped_assessment_log_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Get the ground truth responses
# MAGIC

# COMMAND ----------

suggested_output_col = F.col(f"{_TEXT_ASSESSMENT}.suggested_output")
is_correct_col = F.col(f"{_TEXT_ASSESSMENT}.ratings.answer_correct.value")
# Extract the thumbs up or thumbs down rating and the suggested output
rating_log_df = (
    deduped_assessment_log_df.withColumn("is_correct", is_correct_col)
    .withColumn(
        "suggested_output",
        F.when(suggested_output_col == "", None).otherwise(suggested_output_col),
    )
    .select("request_id", "is_correct", "suggested_output", _RETRIEVAL_ASSESSMENT)
)
# Join the request log with the ratings from above
raw_requests_with_feedback_df = request_log_df.join(
    rating_log_df,
    request_log_df.databricks_request_id == rating_log_df.request_id,
    "left",
)

# COMMAND ----------

# For thumbs up, use either the suggested output or the response, in that order
positive_feedback_df = (
  raw_requests_with_feedback_df
    .where(F.col("is_correct") == F.lit("positive"))
    .withColumn(
      "expected_response",
      F.when(
        F.col("suggested_output") != None, F.col("suggested_output")
      ).otherwise(F.col("response"))
    )
)

# For thumbs down, use the suggested output if there is one
negative_feedback_df = (
  raw_requests_with_feedback_df
    .where(F.col("is_correct") == F.lit("negative"))
    .withColumn("expected_response", F.col("suggested_output"))
)

# For no feedback or IDK, there is no expected response.
no_or_unknown_feedback_df = (
  raw_requests_with_feedback_df
    .where((F.col("is_correct").isNull()) | ((F.col("is_correct") != F.lit("negative")) & (F.col("is_correct") != F.lit("positive"))))
    .withColumn("expected_response", F.lit(None))
)

# COMMAND ----------

# Join the above feedback tables and select the relevant columns for the evaluation
requests_with_feedback_df = positive_feedback_df.unionByName(negative_feedback_df).unionByName(no_or_unknown_feedback_df)
# Get the thumbs up or thumbs down for each retrieved chunk
requests_with_feedback_df = requests_with_feedback_df.withColumn(
    "chunk_at_i_relevance",
    F.transform(
        F.col(_RETRIEVAL_ASSESSMENT),
        lambda x: x.ratings.answer_correct.value
    )
).drop(_RETRIEVAL_ASSESSMENT)
# Convert the PySpark DataFrame to a pandas DataFrame
requests_with_feedback_pdf = requests_with_feedback_df.toPandas()

# COMMAND ----------

def extract_retrieved_chunks_from_trace(trace_str: str) -> List[Mapping[str, str]]:
  """Helper function to extract the retrieved chunks from a trace string"""
  trace = mlflow_entities.Trace.from_json(trace_str)
  chunks = traces.extract_retrieval_context_from_trace(trace)
  return [{"doc_uri": chunk.doc_uri, "content": chunk.content} for chunk in chunks]

def construct_expected_retrieval_context(trace_str: Optional[str], chunk_at_i_relevance: Optional[List[str]]) -> Optional[List[Mapping[str, str]]]:
  """Helper function to construct the expected retrieval context. Any retrieved chunks that are not relevant are dropped."""
  if chunk_at_i_relevance is None or trace_str is None: 
    return None
  retrieved_chunks = extract_retrieved_chunks_from_trace(trace_str)
  return [chunk for chunk, rating in zip(retrieved_chunks, chunk_at_i_relevance) if rating == "true"]

# Construct the expected retrieval context
requests_with_feedback_pdf["expected_retrieved_context"] = requests_with_feedback_pdf.apply(
  lambda row: construct_expected_retrieval_context(row["trace"], row["chunk_at_i_relevance"]), axis=1
)
# Select the columns that are relevant to the evaluation
requests_with_feedback_pdf = requests_with_feedback_pdf[["request", "response", "trace", "expected_response", "expected_retrieved_context", "is_correct"]]
requests_with_feedback_pdf.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Run the evaluation

# COMMAND ----------

result = mlflow.evaluate(
    data=requests_with_feedback_pdf,
    model_type="databricks-agent",
)
result_df = result.tables["eval_results"]

# COMMAND ----------

result_df.display()

# COMMAND ----------


