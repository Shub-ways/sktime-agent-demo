import pandas as pd
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.base import ForecastingHorizon

# sample data
y = pd.Series([10, 20, 30, 40, 50])

# simple "agent"
def parse_query(query):
    if "10" in query:
        return 10
    elif "5" in query:
        return 5
    return 3

query = "predict next 10 days"
steps = parse_query(query)

fh = ForecastingHorizon(range(1, steps + 1), is_relative=True)

model = NaiveForecaster(strategy="last")
model.fit(y)

y_pred = model.predict(fh)

print("Query:", query)
print("Predictions:")
print(y_pred)