# Databricks notebook source
# MAGIC %pip install --upgrade "mlflow-skinny[databricks]>=2.5.0" tensorflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
wind_farm_data = pd.read_csv("https://github.com/dbczumar/model-registry-demo-notebook/raw/master/dataset/windfarm_data.csv", index_col=0)

def get_weather_and_forecast():
  format_date = lambda pd_date : pd_date.date().strftime("%Y-%m-%d")
  today = pd.Timestamp('today').normalize()
  week_ago = today - pd.Timedelta(days=5)
  week_later = today + pd.Timedelta(days=5)

  past_power_output = pd.DataFrame(wind_farm_data)[format_date(week_ago):format_date(today)]
  weather_and_forecast = pd.DataFrame(wind_farm_data)[format_date(week_ago):format_date(week_later)]
  if len(weather_and_forecast) < 10:
    past_power_output = pd.DataFrame(wind_farm_data).iloc[-10:-5]
    weather_and_forecast = pd.DataFrame(wind_farm_data).iloc[-10:]

  return weather_and_forecast.drop(columns="power"), past_power_output["power"]

# COMMAND ----------

def plot(model_name, model_alias, model_version, power_predictions, past_power_output):
  import matplotlib.dates as mdates
  from matplotlib import pyplot as plt
  index = power_predictions.index
  fig = plt.figure(figsize=(11, 7))
  ax = fig.add_subplot(111)
  ax.set_xlabel("Date", size=20, labelpad=20)
  ax.set_ylabel("Power\noutput\n(MW)", size=20, labelpad=60, rotation=0)
  ax.tick_params(axis='both', which='major', labelsize=17)
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
  ax.plot(index[:len(past_power_output)], past_power_output, label="True", color="red", alpha=0.5, linewidth=4)
  ax.plot(index, power_predictions.squeeze(), "--", label="Predicted by '%s'\nwith alias '%s' (Version %d)" % (model_name, model_alias, model_version), color="blue", linewidth=3)
  ax.set_ylim(ymin=0, ymax=max(3500, int(max(power_predictions.values) * 1.3)))
  ax.legend(fontsize=14)
  plt.title("Wind farm power output and projections", size=24, pad=20)
  plt.tight_layout()
  display(plt.show())

# COMMAND ----------

model_uri=f"models:/antonio_test_ws_1.test_model_schema.wind_forecasting@champion"

import mlflow

latest_model = mlflow.pyfunc.load_model(
  model_uri=model_uri
)

weather_data, past_power_output = get_weather_and_forecast()
power_predictions = pd.DataFrame(latest_model.predict(weather_data))
power_predictions.index = pd.to_datetime(weather_data.index)
print(power_predictions)
plot("latest_model", "champion", 1, power_predictions, past_power_output)

# COMMAND ----------

# Define model name in the Model Registry
model_name = f"antonio_test_ws_2.test_model_schema.wind_forecasting"

# Point to Unity-Catalog registry and log/push artifact
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model(
    model_uri=model_uri,
    name=model_name,
)

# COMMAND ----------

model_uri=f"models:/antonio_test_ws_2.test_model_schema.wind_forecasting@champion"

import mlflow

latest_model_2 = mlflow.pyfunc.load_model(
  model_uri=model_uri
)

weather_data, past_power_output = get_weather_and_forecast()
power_predictions = pd.DataFrame(latest_model_2.predict(weather_data))
power_predictions.index = pd.to_datetime(weather_data.index)
print(power_predictions)
plot("latest_model_2", "champion", 1, power_predictions, past_power_output)

# COMMAND ----------


