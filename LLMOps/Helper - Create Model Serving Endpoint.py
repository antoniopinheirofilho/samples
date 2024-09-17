# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install -U --quiet databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Parameters

# COMMAND ----------

model_name = dbutils.widgets.get("model_name")
endpoint_name = dbutils.widgets.get("endpoint_name")
host = dbutils.widgets.get("host")
tracking_table_catalog = dbutils.widgets.get("tracking_table_catalog")
tracking_table_schema = dbutils.widgets.get("tracking_table_schema")
tracking_table_name = dbutils.widgets.get("tracking_table_name")

endpoint_token = dbutils.secrets.get(scope="creds", key="pat")

# COMMAND ----------

# MAGIC %md
# MAGIC # Get latest model version

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

# Point to UC registry
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

latest_model_version = get_latest_model_version(model_name)

if latest_model_version:
  print(f"Model created and logged to: {model_name}/{latest_model_version}")
else:
  raise(BaseException("Error: Model not created, verify if 00-Build-Model script ran successfully!"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy model to a model serving endpoint
# MAGIC - From https://docs.databricks.com/en/_extras/notebooks/source/machine-learning/deploy-mlflow-pyfunc-model-serving.html

# COMMAND ----------

# Name of the registered MLflow model
model_name = model_name

# Get the latest version of the MLflow model
model_version = latest_model_version

# Specify the type of compute (CPU, GPU_SMALL, GPU_MEDIUM, etc.)
workload_type = "GPU_SMALL" 

# Specify the scale-out size of compute (Small, Medium, Large, etc.)
workload_size = "Small" 

# Specify Scale to Zero(only supported for CPU endpoints)
scale_to_zero = False 

# COMMAND ----------

from databricks.sdk.service.serving import EndpointCoreConfigInput

endpoint_config_dict = { 
                        "served_entities": [
                            {
                                "entity_name": model_name,
                                "entity_version": model_version,
                                "workload_size": workload_size,
                                "scale_to_zero_enabled": scale_to_zero,
                                "workload_type": workload_type,
                                "environment_vars": {
                                    "DATABRICKS_HOST": f"{host}",
                                    "DATABRICKS_TOKEN": f"{endpoint_token}"
                                    }
                                }
                            ],
                        "auto_capture_config":{
                            "catalog_name": f"{tracking_table_catalog}",
                            "schema_name": f"{tracking_table_schema}",
                            "table_name_prefix": f"{tracking_table_name}"
                            }
                        }

endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)

print(endpoint_config)

# COMMAND ----------

from databricks.sdk import WorkspaceClient


# Initiate the workspace client
w = WorkspaceClient()
serving_endpoint_name = endpoint_name

# Get endpoint if it exists
existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)

serving_endpoint_url = f"{host}/ml/endpoints/{serving_endpoint_name}"

# If endpoint doesn't exist, create it
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)

# If endpoint does exist, update it to serve the new version
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_entities=endpoint_config.served_entities, name=serving_endpoint_name)

displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')
