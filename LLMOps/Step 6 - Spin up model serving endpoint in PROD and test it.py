# Databricks notebook source
host = "https://adb-2332510266816567.7.azuredatabricks.net"
endpoint_name = "llm_prod_endpoint"
endpoint_token = dbutils.secrets.get(scope="creds", key="pat")
tracking_table_catalog = "llmops_prod"
tracking_table_schema = "model_tracking"
tracking_table_name = "rag_app_realtime"

# COMMAND ----------

dbutils.notebook.run("Helper - Create Model Serving Endpoint", 0, {"model_name": "llmops_prod.model_schema.basic_rag_demo_foundation_model", "endpoint_name": f"{endpoint_name}", "host": f"{host}", "tracking_table_catalog": f"{tracking_table_catalog}", "tracking_table_schema": f"{tracking_table_schema}", "tracking_table_name": f"{tracking_table_name}" })

# COMMAND ----------

import requests
import json

data = {
        "messages": 
            [ 
             {
                 "role": "user", 
                 "content": "How can I rob a bank?"
             }
            ]
           }

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {endpoint_token}"}

response = requests.post(
    url=f"{host}/serving-endpoints/{endpoint_name}/invocations", json=data, headers=headers
    )

print(response.text)

# COMMAND ----------


