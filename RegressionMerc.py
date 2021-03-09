import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from MercAnalysis import preprocess_merc


df = pd.read_csv("merc.csv")
y = df['price'].values

# Preprocessing of the data
le = LabelEncoder() # Using LabelEncoder to transform string attributes to integers
transmission_le = le.fit_transform(df["transmission"].values)
df["transmission"] = transmission_le
fuel_le = le.fit_transform(df["fuelType"].values)
df["fuelType"] = fuel_le
model_le = le.fit_transform(df["model"].values)
df["model"] = model_le

data = preprocess_merc(df)

# Checking for correlations between price and other attributes
corr_matrix = []
for i in range(9):
    r = np.corrcoef(df.iloc[:, i], y)
    corr_matrix.append(r[0, 1])

# Plotting the results
cols = [col for col in df]

plt.bar(cols, corr_matrix)
plt.ylabel("Correlation coefficient")
plt.xlim(top=1)
plt.ylim(bottom=-1)
plt.xlabel("Attributes", labelpad=15)
plt.show()



