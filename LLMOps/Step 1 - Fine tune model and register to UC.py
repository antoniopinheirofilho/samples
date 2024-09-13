# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

# Let's start by installing our products
%pip install databricks-genai==1.0.2
%pip install databricks-sdk==0.27.1
%pip install "mlflow==2.12.2"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Create datasets for fine-tunning

# COMMAND ----------

# DBTITLE 1,Helper classes
import requests
import collections
import os


class DBDemos():
  @staticmethod
  def setup_schema(catalog, db, reset_all_data, volume_name = None):
    if reset_all_data:
      print(f'clearing up volume named `{catalog}`.`{db}`.`{volume_name}`')
      spark.sql(f"DROP VOLUME IF EXISTS `{catalog}`.`{db}`.`{volume_name}`")
      spark.sql(f"DROP SCHEMA IF EXISTS `{catalog}`.`{db}` CASCADE")

    def use_and_create_db(catalog, dbName, cloud_storage_path = None):
      print(f"USE CATALOG `{catalog}`")
      spark.sql(f"USE CATALOG `{catalog}`")
      spark.sql(f"""create database if not exists `{dbName}` """)

    assert catalog not in ['hive_metastore', 'spark_catalog'], "This demo only support Unity. Please change your catalog name."
    #If the catalog is defined, we force it to the given value and throw exception if not.
    current_catalog = spark.sql("select current_catalog()").collect()[0]['current_catalog()']
    if current_catalog != catalog:
      catalogs = [r['catalog'] for r in spark.sql("SHOW CATALOGS").collect()]
      if catalog not in catalogs:
        spark.sql(f"CREATE CATALOG IF NOT EXISTS `{catalog}`")
        if catalog == 'dbdemos':
          spark.sql(f"ALTER CATALOG `{catalog}` OWNER TO `account users`")
    use_and_create_db(catalog, db)

    if catalog == 'dbdemos':
      try:
        spark.sql(f"GRANT CREATE, USAGE on DATABASE `{catalog}`.`{db}` TO `account users`")
        spark.sql(f"ALTER SCHEMA `{catalog}`.`{db}` OWNER TO `account users`")
        for t in spark.sql(f'SHOW TABLES in {catalog}.{db}').collect():
          try:
            spark.sql(f'GRANT ALL PRIVILEGES ON TABLE {catalog}.{db}.{t["tableName"]} TO `account users`')
            spark.sql(f'ALTER TABLE {catalog}.{db}.{t["tableName"]} OWNER TO `account users`')
          except Exception as e:
            if "NOT_IMPLEMENTED.TRANSFER_MATERIALIZED_VIEW_OWNERSHIP" not in str(e) and "STREAMING_TABLE_OPERATION_NOT_ALLOWED.UNSUPPORTED_OPERATION" not in str(e) :
              print(f'WARN: Couldn t set table {catalog}.{db}.{t["tableName"]} owner to account users, error: {e}')
      except Exception as e:
        print("Couldn't grant access to the schema to all users:"+str(e))    

    print(f"using catalog.database `{catalog}`.`{db}`")
    spark.sql(f"""USE `{catalog}`.`{db}`""")    

    if volume_name:
      spark.sql(f'CREATE VOLUME IF NOT EXISTS {volume_name};')

                     
  #Return true if the folder is empty or does not exists
  @staticmethod
  def is_folder_empty(folder):
    try:
      return len(dbutils.fs.ls(folder)) == 0
    except:
      return True
    
  @staticmethod
  def is_any_folder_empty(folders):
    return any([DBDemos.is_folder_empty(f) for f in folders])

  @staticmethod
  def set_model_permission(model_name, permission, principal):
    import databricks.sdk.service.catalog as c
    sdk_client = databricks.sdk.WorkspaceClient()
    return sdk_client.grants.update(c.SecurableType.FUNCTION, model_name, changes=[
                              c.PermissionsChange(add=[c.Privilege[permission]], principal=principal)])

  @staticmethod
  def set_model_endpoint_permission(endpoint_name, permission, group_name):
    import databricks.sdk.service.serving as s
    sdk_client = databricks.sdk.WorkspaceClient()
    ep = sdk_client.serving_endpoints.get(endpoint_name)
    return sdk_client.serving_endpoints.set_permissions(serving_endpoint_id=ep.id, access_control_list=[s.ServingEndpointAccessControlRequest(permission_level=s.ServingEndpointPermissionLevel[permission], group_name=group_name)])

  @staticmethod
  def set_index_permission(index_name, permission, principal):
      import databricks.sdk.service.catalog as c
      sdk_client = databricks.sdk.WorkspaceClient()
      return sdk_client.grants.update(c.SecurableType.TABLE, index_name, changes=[
                              c.PermissionsChange(add=[c.Privilege[permission]], principal=principal)])
    

  @staticmethod
  def download_file_from_git(dest, owner, repo, path):
    def download_file(url, destination):
      local_filename = url.split('/')[-1]
      # NOTE the stream=True parameter below
      with requests.get(url, stream=True) as r:
        r.raise_for_status()
        print('saving '+destination+'/'+local_filename)
        with open(destination+'/'+local_filename, 'wb') as f:
          for chunk in r.iter_content(chunk_size=8192): 
            # If you have chunk encoded response uncomment if
            # and set chunk_size parameter to None.
            #if chunk: 
            f.write(chunk)
      return local_filename

    if not os.path.exists(dest):
      os.makedirs(dest)
    from concurrent.futures import ThreadPoolExecutor
    files = requests.get(f'https://api.github.com/repos/{owner}/{repo}/contents{path}').json()
    files = [f['download_url'] for f in files if 'NOTICE' not in f['name']]
    def download_to_dest(url):
      try:
        #Temporary fix to avoid hitting github limits - Swap github to our S3 bucket to download files
        s3url = url.replace("https://raw.githubusercontent.com/databricks-demos/dbdemos-dataset/main/", "https://notebooks.databricks.com/demos/dbdemos-dataset/")
        download_file(s3url, dest)
      except:
        download_file(url, dest)
    with ThreadPoolExecutor(max_workers=10) as executor:
      collections.deque(executor.map(download_to_dest, files))
         

  #force the experiment to the field demos one. Required to launch as a batch
  @staticmethod
  def init_experiment_for_batch(demo_name, experiment_name):
    import mlflow
    #You can programatically get a PAT token with the following
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    xp_root_path = f"/Shared/dbdemos/experiments/{demo_name}"
    try:
      r = w.workspace.mkdirs(path=xp_root_path)
    except Exception as e:
      print(f"ERROR: couldn't create a folder for the experiment under {xp_root_path} - please create the folder manually or  skip this init (used for job only: {e})")
      raise e
    xp = f"{xp_root_path}/{experiment_name}"
    print(f"Using common experiment under {xp}")
    mlflow.set_experiment(xp)
    DBDemos.set_experiment_permission(xp)
    return mlflow.get_experiment_by_name(xp)

  @staticmethod
  def set_experiment_permission(experiment_path):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service import iam
    w = WorkspaceClient()
    try:
      status = w.workspace.get_status(experiment_path)
      w.permissions.set("experiments", request_object_id=status.object_id,  access_control_list=[
                            iam.AccessControlRequest(group_name="users", permission_level=iam.PermissionLevel.CAN_MANAGE)])    
    except Exception as e:
      print(f"error setting up shared experiment {experiment_path} permission: {e}")

    print(f"Experiment on {experiment_path} was set public")


  @staticmethod
  def get_active_streams(start_with = ""):
    return [s for s in spark.streams.active if len(start_with) == 0 or (s.name is not None and s.name.startswith(start_with))]

  @staticmethod
  def stop_all_streams_asynch(start_with = "", sleep_time=0):
    import threading
    def stop_streams():
        DBDemos.stop_all_streams(start_with=start_with, sleep_time=sleep_time)

    thread = threading.Thread(target=stop_streams)
    thread.start()

  @staticmethod
  def stop_all_streams(start_with = "", sleep_time=0):
    import time
    time.sleep(sleep_time)
    streams = DBDemos.get_active_streams(start_with)
    if len(streams) > 0:
      print(f"Stopping {len(streams)} streams")
      for s in streams:
          try:
              s.stop()
          except:
              pass
      print(f"All stream stopped {'' if len(start_with) == 0 else f'(starting with: {start_with}.)'}")

  @staticmethod
  def wait_for_all_stream(start = ""):
    import time
    actives = DBDemos.get_active_streams(start)
    if len(actives) > 0:
      print(f"{len(actives)} streams still active, waiting... ({[s.name for s in actives]})")
    while len(actives) > 0:
      spark.streams.awaitAnyTermination()
      time.sleep(1)
      actives = DBDemos.get_active_streams(start)
    print("All streams completed.")

  @staticmethod
  def get_last_experiment(demo_name, experiment_path = "/Shared/dbdemos/experiments/"):
    import requests
    import re
    from datetime import datetime
    #TODO: waiting for https://github.com/databricks/databricks-sdk-py/issues/509 to use the python sdk instead
    base_url =dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    r = requests.get(base_url+"/api/2.0/workspace/list", params={'path': f"{experiment_path}/{demo_name}"}, headers=headers).json()
    if 'objects' not in r:
      raise Exception(f"No experiment available for this demo. Please re-run the previous notebook with the AutoML run. - {r}")
    xps = [f for f in r['objects'] if f['object_type'] == 'MLFLOW_EXPERIMENT' and 'automl' in f['path']]
    xps = [x for x in xps if re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})', x['path'])]
    sorted_xp = sorted(xps, key=lambda f: f['path'], reverse = True)
    if len(sorted_xp) == 0:
      raise Exception(f"No experiment available for this demo. Please re-run the previous notebook with the AutoML run. - {r}")

    last_xp = sorted_xp[0]

    # Search for the date pattern in the input string
    match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})', last_xp['path'])

    if match:
        date_str = match.group(1)  # Extract the matched date string
        date = datetime.strptime(date_str, '%Y-%m-%d_%H:%M:%S')  # Convert to a datetime object
        # Calculate the difference in days from the current date
        days_difference = (datetime.now() - date).days
        if days_difference > 30:
            raise Exception(f"It looks like the last experiment {last_xp} is too old ({days} days). Please re-run the previous notebook to make sure you have the latest version.")
    else:
        raise Exception(f"Invalid experiment format or no experiment available. Please re-run the previous notebook. {last_xp['path']}")
    return last_xp

# COMMAND ----------

catalog_training_data = "demo_prep"
schema_training_data = "fine_tunning"
volume_folder =  f"/Volumes/{catalog_training_data}/{schema_training_data}/git_files"

# COMMAND ----------

if not spark.catalog.tableExists(f'{catalog_training_data}.{schema_training_data}.training_dataset_question') or \
    not spark.catalog.tableExists(f'{catalog_training_data}.{schema_training_data}.training_dataset_answer') or \
    not spark.catalog.tableExists(f'{catalog_training_data}.{schema_training_data}.databricks_documentation')or \
    not spark.catalog.tableExists(f'{catalog_training_data}.{schema_training_data}.customer_tickets'):
      
  DBDemos.download_file_from_git(volume_folder+"/training_dataset", "databricks-demos", "dbdemos-dataset", "llm/databricks-documentation")

  #spark.read.format('parquet').load(f"{volume_folder}/training_dataset/raw_documentation.parquet").write.saveAsTable("raw_documentation")
  spark.read.format('parquet').load(f"{volume_folder}/training_dataset/training_dataset_question.parquet").write.mode('overwrite').saveAsTable(f"{catalog_training_data}.{schema_training_data}.training_dataset_question")
  spark.read.format('parquet').load(f"{volume_folder}/training_dataset/training_dataset_answer.parquet").write.mode('overwrite').saveAsTable(f"{catalog_training_data}.{schema_training_data}.training_dataset_answer")
  spark.read.format('parquet').load(f"{volume_folder}/training_dataset/databricks_documentation.parquet").write.mode('overwrite').saveAsTable(f"{catalog_training_data}.{schema_training_data}.databricks_documentation")
  spark.read.format('parquet').load(f"{volume_folder}/training_dataset/customer_tickets.parquet").write.mode('overwrite').saveAsTable(f"{catalog_training_data}.{schema_training_data}.customer_tickets")

# COMMAND ----------

training_dataset = spark.sql(f"""
  SELECT q.id as question_id, q.question, a.answer, d.url, d.content 
  FROM {catalog_training_data}.{schema_training_data}.training_dataset_question q
      INNER JOIN {catalog_training_data}.{schema_training_data}.databricks_documentation d on q.doc_id = d.id
      INNER JOIN {catalog_training_data}.{schema_training_data}.training_dataset_answer   a on a.question_id = q.id 
    WHERE answer IS NOT NULL""")
    
display(training_dataset)

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd

#base_model_name = "meta-llama/Llama-2-7b-hf"
#base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
base_model_name = "meta-llama/Llama-2-70b-chat-hf"

system_prompt = """You are a highly knowledgeable and professional Databricks Support Agent. Your goal is to assist users with their questions and issues related to Databricks. Answer questions as precisely and accurately as possible, providing clear and concise information. If you do not know the answer, respond with "I don't know." Be polite and professional in your responses. Provide accurate and detailed information related to Databricks. If the question is unclear, ask for clarification.\n"""

@pandas_udf("array<struct<role:string, content:string>>")
def create_conversation(content: pd.Series, question: pd.Series, answer: pd.Series) -> pd.Series:
    def build_message(c,q,a):
        user_input = f"Here is a documentation page that could be relevant: {c}. Based on this, answer the following question: {q}"
        if "mistral" in base_model_name:
            #Mistral doesn't support system prompt
            return [
                {"role": "user", "content": f"{system_prompt} \n{user_input}"},
                {"role": "assistant", "content": a}]
        else:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": a}]
    return pd.Series([build_message(c,q,a) for c, q, a in zip(content, question, answer)])


training_data, eval_data = training_dataset.randomSplit([0.9, 0.1], seed=42)

training_data.select(create_conversation("content", "question", "answer").alias('messages')).write.mode('overwrite').saveAsTable(f"{catalog_training_data}.{schema_training_data}.chat_completion_training_dataset")
eval_data.write.mode('overwrite').saveAsTable(f"{catalog_training_data}.{schema_training_data}.chat_completion_evaluation_dataset")

display(spark.table(f'{catalog_training_data}.{schema_training_data}.chat_completion_training_dataset'))

# COMMAND ----------

# MAGIC %md
# MAGIC # Fine-tune the foundation model to use in a RAG app

# COMMAND ----------

catalog_model = "demo_prep"
schema_model = "fine_tunning"

# COMMAND ----------

import re

from databricks.model_training import foundation_model as fm
#Return the current cluster id to use to read the dataset and send it to the fine tuning cluster. See https://docs.databricks.com/en/large-language-models/foundation-model-training/create-fine-tune-run.html#cluster-id
def get_current_cluster_id():
  import json
  return json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().safeToJson())['attributes']['clusterId']


#Let's clean the model name
registered_model_name = f"{catalog_model}.{schema_model}.fine_tuned_" + re.sub(r'[^a-zA-Z0-9]', '_',  base_model_name)

run = fm.create(
    data_prep_cluster_id=get_current_cluster_id(),  # required if you are using delta tables as training data source. This is the cluster id that we want to use for our data prep job.
    model=base_model_name,  # Here we define what model we used as our baseline
    train_data_path=f"{catalog_training_data}.{schema_training_data}.chat_completion_training_dataset",
    task_type="CHAT_COMPLETION",  # Change task_type="INSTRUCTION_FINETUNE" if you are using the fine-tuning API for completion.
    register_to=registered_model_name,
    training_duration="1ep", #only 5 epoch to accelerate the demo. Check the mlflow experiment metrics to see if you should increase this number
    learning_rate="5e-7",
)

print(run)

# COMMAND ----------

#Helper fuinction to Wait for the fine tuning run to finish
def wait_for_run_to_finish(run):
  import time
  for i in range(300):
    events = run.get_events()
    for e in events:
      if "FAILED" in e.type or "EXCEPTION" in e.type:
        raise Exception(f'Error with the fine tuning run, check the details in run.get_events(): {e}')
    if events[-1].type == 'COMPLETED':
      print('Run finished')
      display(events)
      return events
    if i % 30 == 0:
      print(f'waiting for run {run.name} to complete...')
    time.sleep(10)

# COMMAND ----------

displayHTML(f'Open the <a href="/ml/experiments/{run.experiment_id}/runs/{run.run_id}/model-metrics">training run on MLFlow</a> to track the metrics')
#Track the run details
display(run.get_events())

#helper function waiting on the run to finish - see the _resources folder for more details
wait_for_run_to_finish(run)

# COMMAND ----------


