import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.ticker


df = pd.read_csv("merc.csv")  # Loading the dataset

le = LabelEncoder() # Using LabelEncoder to transform string attributes to integers
transmission_le = le.fit_transform(df["transmission"].values)
df["transmission"] = transmission_le
fuel_le = le.fit_transform(df["fuelType"].values)
df["fuelType"] = fuel_le
model_le = le.fit_transform(df["model"].values)
df["model"] = model_le

# Checking the data
info = df.info()
engines = df.engineSize.value_counts()  # 12 engines of size 0.0
transmission = df.transmission.value_counts()  # everything seems to be normal
year = df.year.value_counts()  # Get rid off 1970, 1999, 1997, 1998 to get better results
mileage = df.mileage.value_counts()  # good
fuel = df.fuelType.value_counts()  # good
tax = df.tax.value_counts()  # good, car tax can be equal to 0
mpg = df.mpg.value_counts()  # good


def preprocess_mercedes(data):
    """
    This function erases data points which I have found invalid during visual
    and numerical data analysis.
    """
    invalid_sizes = data[data["engineSize"] == 0.0].index
    for index in invalid_sizes:
        data = data.drop(index, axis=0)

    old_years = data[data["year"] < 2000].index
    for index in old_years:
        data = data.drop(index, axis=0)

    invalid_combustion = data[data["mpg"] < 10].index
    for index in invalid_combustion:
        data = data.drop(index, axis=0)

    data = data.reset_index(drop=True)

    return data


# Preprocessing the data before visualization
df = preprocess_mercedes(df)
y = df['price'].values

# Creating a correlation bar graph
corr_matrix = []
for i in range(9):
    r = np.corrcoef(df.iloc[:, i], y)
    corr_matrix.append(r[0, 1])

# Plotting the results
cols = [col for col in df]

plt.bar(cols, corr_matrix)
plt.ylabel("Correlation coefficient")
plt.ylim(top=1)
plt.ylim(bottom=-1)
plt.xlabel("Attributes", labelpad=15)
plt.show()
# I've chosen to investigate on year, model, mileage, mpg and engineSize


def visualise():
    """
    Function which visualize relationships between attributes and a price.

    """
    plt.scatter(df["model"].values, y)
    plt.title("Price/Model relationship")
    plt.ylabel("Price [£]")
    plt.xlabel("Model encoded value")
    plt.show()

    plt.scatter(df['year'].values, y)  # Exponential behaviour
    plt.title("Price/Year relationship")
    plt.ylabel("Price [£]")
    plt.xlabel("Year")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()

    plt.scatter(df["mileage"], y)  # Decreasing exponential behaviour with one strange data point
    plt.title("Price/Mileage relationship")
    plt.ylabel("Price [£]")
    plt.xlabel("Mileage [miles]")
    plt.show()

    plt.scatter(df['mpg'].values, y)  # Kind off decreasing exponential but not to accurate scatter
    plt.title("Price/Combustion relationship")
    plt.ylabel("Price [£]")
    plt.xlabel("Combustion [miles per gallon]")
    plt.show()

    plt.scatter(df['engineSize'], y)
    plt.title("Price/Engine size relationship")
    plt.ylabel("Price [£]")
    plt.xlabel("Engine size")
    plt.show()

