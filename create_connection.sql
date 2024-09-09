-- Databricks notebook source
CREATE CONNECTION snow_secret TYPE snowflake
OPTIONS (
  host 'yhksbiy-ira76331.snowflakecomputing.com',
  port '443',
  user 'leonildopfilho',
  sfWarehouse 'COMPUTE_WH',
  password secret ('sql_conn','pwd_snow')
)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC
-- MAGIC scr = dbutils.secrets.get(scope = "sql_conn", key = "pwd_snow")
-- MAGIC
-- MAGIC for s in scr:
-- MAGIC     print(s)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC
-- MAGIC import requests
-- MAGIC
-- MAGIC pwd = dbutils.secrets.get(scope = "snowflake_secret_scope_2", key = "snowflake_pwd")
-- MAGIC
-- MAGIC api_url = "https://adb-53096060768276.16.azuredatabricks.net/api/2.1/unity-catalog/connections/snow_secret_terraform"
-- MAGIC
-- MAGIC payload = {
-- MAGIC   "options": {
-- MAGIC     "host": "yhksbiy-ira76331.snowflakecomputing.com",
-- MAGIC     "port": "443",
-- MAGIC     "user": "leonildopfilho",
-- MAGIC     "sfWarehouse": "COMPUTE_WH",
-- MAGIC     "password": pwd
-- MAGIC   }
-- MAGIC }
-- MAGIC
-- MAGIC headers = {
-- MAGIC     'Content-Type': 'application/json',
-- MAGIC     'Authorization': 'Bearer XXXXX'
-- MAGIC }
-- MAGIC
-- MAGIC response = requests.request("PATCH", api_url, headers=headers, json=payload)
-- MAGIC
-- MAGIC print(response.status_code)
-- MAGIC print(response.text)

-- COMMAND ----------


