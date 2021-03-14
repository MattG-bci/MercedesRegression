import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from MercAnalysis import preprocess_mercedes


df = pd.read_csv("merc.csv")

# Preprocessing of the data
le = LabelEncoder() # Using LabelEncoder to transform string attributes to integers
transmission_le = le.fit_transform(df["transmission"].values)
df["transmission"] = transmission_le
fuel_le = le.fit_transform(df["fuelType"].values)
df["fuelType"] = fuel_le
model_le = le.fit_transform(df["model"].values)
df["model"] = model_le

data = preprocess_mercedes(df)
prices = df['price'].values
attributes = [df['year'].values, df['mileage'].values, df['mpg'].values]

x_train, x_test, y_train, y_test = train_test_split(df[['year', 'mileage', 'mpg', 'model', 'fuelType', 'transmission', 'engineSize']], prices, test_size=0.2)

reg = LinearRegression()
reg.fit(x_train, y_train)
reg.predict(x_test)
print(reg.score(x_test, y_test))

