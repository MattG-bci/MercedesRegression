import pandas as pd

df = pd.read_csv("merc.csv")

# Checking the data
#info = df.info()
engines = df.engineSize.value_counts()  # 12 engines of size 0.0
transmission = df.transmission.value_counts()  # everything seems to be normal
year = df.year.value_counts()  # Get rid off 1970, 1999, 1997, 1998 to get better results
mileage = df.mileage.value_counts()  # good
fuel = df.fuelType.value_counts()  # good
tax = df.tax.value_counts()  # good, car tax can be equal to 0
mpg = df.mpg.value_counts()  # good


def preprocess_merc(df):
    errors = df[df["engineSize"] == 0.0].index
    for index in errors:
        df = df.drop(index, axis=0)

    old_years = df[df["year"] < 2000].index
    for index in old_years:
        df = df.drop(index, axis=0)

    df = df.reset_index(drop=True)

    return df

