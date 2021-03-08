import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("merc.csv")

#info = df.info()
engines = df.engineSize.value_counts()  # 12 engines of size 0.0
transmission = df.transmission.value_counts()  # everything seems to be normal
year = df.year.value_counts()  # Get rid off 1970, 1999, 1997, 1998 to get better results
mileage = df.mileage.value_counts()  # good
fuel = df.fuelType.value_counts()  # good
tax = df.tax.value_counts()  # good, car tax can be equal to 0
mpg = df.mpg.value_counts()  # good
