# Databricks notebook source
dbutils.notebook.run("Helper - Create Model Serving Endpoint", 60, {"model_name": "llmops_dev.model_schema.basic_rag_demo_foundation_model", "endpoint_name": "llm_validation_endpoint", "host": "https://adb-2332510266816567.7.azuredatabricks.net" })

# COMMAND ----------


