import pandas as pd
import numpy as np



# Load data
data = pd.read_csv('C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 02\\15. TimeSeries\\climate.csv')
# print(data.shape)


data = data.drop("Date Time", axis = 1)
#print(data.head())


# Function to extract sequences and target variable from given data
def pairing(data, seq_len, target_name):

    feature_target = []
    target = []
    for i in range(0, data.shape[0] - (seq_len + 1), seq_len + 1):

         feature_target.append(data[i: seq_len + i])
         target.append(data[target_name][seq_len + i])

    return np.array(feature_target), np.array(target)

x, y = pairing(data, 6, "T (degC)")

#print(x.shape)
#print(y.shape)
#print(x[:3])





