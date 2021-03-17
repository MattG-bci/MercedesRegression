import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from MercAnalysis import preprocess_mercedes


df = pd.read_csv("merc.csv")  # Loading the data

# Preprocessing the data
le = LabelEncoder()  # Using LabelEncoder to transform string attributes to integers
transmission_le = le.fit_transform(df["transmission"].values)
df["transmission"] = transmission_le
fuel_le = le.fit_transform(df["fuelType"].values)
df["fuelType"] = fuel_le
model_le = le.fit_transform(df["model"].values)
df["model"] = model_le

data = preprocess_mercedes(df)  # imported function which removes invalid data from dataset
prices = df['price'].values

x_train, x_test, y_train, y_test = train_test_split(df[['year', 'mileage', 'mpg', 'model', 'fuelType',
                                                        'transmission', 'engineSize']], prices, test_size=0.2)

# Training the model and predicting the values
reg = LinearRegression()
reg.fit(x_train, y_train)
reg.predict(x_test)
summary = np.array([reg.predict(x_test), y_test, reg.score(x_test, y_test)], dtype=object)  # Creating an array for summary
print(summary)

