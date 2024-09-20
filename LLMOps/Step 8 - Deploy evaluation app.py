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

# MAGIC %md
# MAGIC # Parameters

# COMMAND ----------

model_name = "llmops_prod.model_schema.basic_rag_demo_foundation_model"
host = "https://adb-2332510266816567.7.azuredatabricks.net"
endpoint_token = dbutils.secrets.get(scope="creds", key="pat")

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy app

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")

def get_latest_model_version(model_name_in:str = None):
    """
    Get latest version of registered model
    """
    client = MlflowClient()
    model_version_infos = client.search_model_versions("name = '%s'" % model_name_in)

    if model_version_infos:
      return max([int(model_version_info.version) for model_version_info in model_version_infos])
    else:
      return None

# COMMAND ----------

latest_version = get_latest_model_version(model_name)

print(latest_version)

# COMMAND ----------

from databricks.agents import deploy
from mlflow.utils import databricks_utils as du

deploy(model_name, 
       latest_version, 
       environment_vars={
           "DATABRICKS_HOST": f"{host}",
           "DATABRICKS_TOKEN": f"{endpoint_token}"
           })

# COMMAND ----------


