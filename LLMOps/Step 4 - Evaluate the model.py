# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

# MAGIC %pip install mlflow databricks-sdk evaluate rouge_score databricks-agents

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Parameters

# COMMAND ----------

endpoint_name = "llm_validation_endpoint"
endpoint_host = "https://adb-3630608912046230.10.azuredatabricks.net"
endpoint_token = ""
inference_table = "llmops_dev.model_tracking.rag_app_realtime_payload"
eval_table = "demo_prep.fine_tunning.chat_completion_evaluation_dataset"

# COMMAND ----------

# MAGIC %md
# MAGIC # Test endpoint
# MAGIC - TODO: MLFlow and SDK didn't accept this model signature for some reason. That is why I'm using the API. It is probably because the model signature doesn't adhere to the openAI standards
# MAGIC - TODO: Is it better to evaluate using MLFlow without a model serving endpoint? How can we parallelize this inference?

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

    return response.json()[0]

# COMMAND ----------

response = get_answer("How do I create a DataFrame using PySpark in Databricks and load data from a CSV file into it?")

# COMMAND ----------

docs = response["context"]

retrieved_context = []
for d in docs:

    id = d["metadata"]["id"]
    content = d["page_content"]

    retrieved_context.append({f"doc_uri": f"{id}", "content": f"{content}"})

print(retrieved_context)

# COMMAND ----------

print(docs)

# COMMAND ----------

response["answer"]

# COMMAND ----------

# MAGIC %md
# MAGIC # Inference table

# COMMAND ----------

display(spark.table(f"{inference_table}"))

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC # Eval Dataset

# COMMAND ----------

df_eval = spark.table(eval_table).limit(5)
display(df_eval)

# COMMAND ----------

# MAGIC %md
# MAGIC # Build data set for evaluation

# COMMAND ----------

request_id = 0

ds_for_eval = []
for row in df_eval.collect():

    request_id += 1
    question = row["question"]
    expected_answer = row["answer"]

    response = get_answer(question)

    answer = response["answer"]

    docs = response["context"]
    retrieved_context = []
    for d in docs:

        id = d["metadata"]["id"]
        content = d["page_content"]

        retrieved_context.append({f"doc_uri": f"{id}", "content": f"{content}"})

    ds_for_eval.append({"request": f"{question}", "response": f"{answer}", "retrieved_context": retrieved_context, "expected_response": f"{expected_answer}"})

print(ds_for_eval)

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaludate dataset using the Databricks Mosaic Agent (LLM as a judge)

# COMMAND ----------

import mlflow
import pandas as pd

pd_data = pd.DataFrame(ds_for_eval)

pd_data.display()

# COMMAND ----------

result = mlflow.evaluate(
    data=pd_data,
    model_type="databricks-agent",
)

display(result.tables['eval_results'])

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaludate dataset using MLflow evaluators

# COMMAND ----------

pd_data_mlflow = pd_data.rename(columns={'request': 'inputs', 'expected_response': 'ground_truth', 'response': 'predictions'})
pd_data_mlflow = pd_data_mlflow[['inputs', 'ground_truth', 'predictions']]

pd_data_mlflow.display()

# COMMAND ----------

with mlflow.start_run() as run:
    results = mlflow.evaluate(
        data=pd_data_mlflow,
        targets="ground_truth",
        predictions="predictions",
        extra_metrics=[mlflow.metrics.genai.answer_similarity()],
        evaluators="default"
    )
    print(f"See aggregated evaluation results below: \n{results.metrics}")

    eval_table = results.tables["eval_results_table"]
    print(f"See evaluation table below: \n{eval_table}")

# COMMAND ----------

with mlflow.start_run() as run:
    results = mlflow.evaluate(
        data=pd_data_mlflow,
        targets="ground_truth",
        predictions="predictions",
        model_type="question-answering"
    )
    
    results.tables["eval_results_table"].display()

# COMMAND ----------

with mlflow.start_run() as run:
    results = mlflow.evaluate(
        data=pd_data_mlflow,
        targets="ground_truth",
        predictions="predictions",
        evaluators="default"
    )
    
    results.tables["eval_results_table"].display()

# COMMAND ----------

# Check https://huggingface.co/spaces/evaluate-metric/bleu
# Check https://huggingface.co/spaces/evaluate-metric/rouge