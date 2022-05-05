
import pandas as pd
import numpy as np


# Load data
data = pd.read_csv('C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 02\\15. TimeSeries\\climate.csv')
# print(data.shape)

data = data.drop("Date Time", axis = 1)
#print(data.head())


# Function to extract sequences and target variable from given data
def pairing(data, seq_len = 10):

    x = []
    y = []
    for i in range(0, (data.shape[0] - seq_len + 1), seq_len + 1):

        seq = np.zeros((seq_len, data.shape[1]))

        for j in range(seq_len):
            seq[j] = data.values[i + j]
            

        x.append(seq)
        y.append(data['T (degC)'][i + seq_len])

    return np.array(x), np.array(y)


pairing(data.head())

# Extract chunks (sequence) of data and the target variable
x, y = pairing(data)


print(x.shape)
print(y.shape)
