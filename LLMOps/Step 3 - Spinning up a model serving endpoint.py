# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

# MAGIC %pip install -U --quiet langchain langchain-community databricks-vectorsearch pydantic mlflow  databricks-sdk cloudpickle "unstructured[pdf,docx]==0.10.30"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Get latest model version

# COMMAND ----------

# Model URI

model_name = "llmops_dev.model_schema.basic_rag_demo"

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
      return max([model_version_info.version for model_version_info in model_version_infos])
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

# Set the name of the MLflow endpoint
endpoint_name = "llm_validation_endpoint_fine_tuned"

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

# Get the API endpoint and token for the current notebook context
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() 
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

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
                                    "DATABRICKS_HOST": "https://adb-3630608912046230.10.azuredatabricks.net",
                                    "DATABRICKS_TOKEN": ""
                                    }
                                }
                            ],
                        "auto_capture_config":{
                            "catalog_name": "llmops_dev",
                            "schema_name": "model_tracking",
                            "table_name_prefix": "rag_app_realtime_fine_tuned"
                            }
                        }

endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)

# COMMAND ----------

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

db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName").value()
serving_endpoint_url = f"{db_host}/ml/endpoints/{serving_endpoint_name}"

# If endpoint doesn't exist, create it
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)

# If endpoint does exist, update it to serve the new version
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_entities=endpoint_config.served_entities, name=serving_endpoint_name)

displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')
