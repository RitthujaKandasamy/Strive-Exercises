import pandas as pd


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


# Assign colum names to the dataset
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']


# Read dataset to pandas dataframe
irisdata = pd.read_csv(url, names=colnames)
print(irisdata.head())
print("irisdata shape: {} \n".format(irisdata.shape))
print(irisdata.info())


# checking for duplicated rows
duplicated_data = irisdata.duplicated()
print("\n Total duplicated values: {}".format(duplicated_data.sum()))
print(irisdata[duplicated_data])

# map target
mapper = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
irisdata['Class'] = irisdata['Class'].map(mapper)


# create X,y
X = irisdata.drop(['Class'], axis = 1)
y = irisdata['Class']
print("\n Feature shape: {}, Target shape: {} \n".format(X.shape, y.shape))


# number of unique values
unique_values = []
for i in irisdata.columns:
    unique_count = irisdata[i].value_counts().count()
    unique_values.append([unique_count])
unique_data = pd.DataFrame(unique_values, index = irisdata.columns, columns = ['Unique values count'] )
print(unique_data)


# take categorial and numerical data for pipeline usage        
num_vars  = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
