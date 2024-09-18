# Databricks notebook source
host = "https://adb-2332510266816567.7.azuredatabricks.net"
endpoint_name = "llm_validation_endpoint"
endpoint_token = dbutils.secrets.get(scope="creds", key="pat")
tracking_table_catalog = "llmops_dev"
tracking_table_schema = "model_tracking"
tracking_table_name = "rag_app_realtime"

# COMMAND ----------

dbutils.notebook.run("Helper - Create Model Serving Endpoint", 0, {"model_name": "llmops_dev.model_schema.basic_rag_demo_foundation_model", "endpoint_name": f"{endpoint_name}", "host": f"{host}", "tracking_table_catalog": f"{tracking_table_catalog}", "tracking_table_schema": f"{tracking_table_schema}", "tracking_table_name": f"{tracking_table_name}" })

# COMMAND ----------


