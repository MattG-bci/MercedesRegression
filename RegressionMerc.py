import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("merc.csv")
df['transmission'] = df['transmission'].replace(['Automatic', 'Semi-Auto', 'Manual', 'Other'], [3, 2, 1, 0])
df['fuelType'] = df['fuelType'].replace(['Other', 'Petrol', 'Diesel', 'Hybrid'], [0, 1, 2, 3])
y = df['price'].values




#seek = {}
#for model in df.iloc[:, 0]:
    #if model not in seek:
    #    seek[model] = 1
   # else:
     #   seek[model] += 1

#models = {key: val for key, val in sorted(seek.items(), key=lambda item: item[1])}

#for index, model in enumerate(seek):
    #df["model"] = df["model"].replace(model, index)

corr_matrix = []
for i in range(1, 9):
    r = np.corrcoef(df.iloc[:, i], y)
    corr_matrix.append(r[0, 1])

cols = [col for col in df]

#plt.bar(cols, corr_matrix)
#plt.ylabel("Correlation coefficient")
#plt.ylim(bottom=-1)
#plt.xlabel("Attributes", labelpad=15)
#plt.show()

# Read about sklearn.preprocessing.LinearEncoder
# Check data for some non-sense data (ex. engineSize = 0.0)


