import pandas as pd


# Read dataset to pandas dataframe
data = pd.read_csv("C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 02\\09. Challenge\\kaggle\\train.csv\\train.csv")
# print(data.head())
# print("data shape: {} \n".format(data.shape))
# print(data.info())


# checking for duplicated rows
duplicated_data = data.duplicated()
# print("\n Total duplicated values: {}".format(duplicated_data.sum()))
# print(data[duplicated_data])


# drop f_04, f_03, f_12, f_06, f_17, f_07, f_18
# these features are not having good correlation


# create X, y
X = data.drop(['target', 'id', 'f_04', 'f_03', 'f_12', 'f_06', 'f_17', 'f_07', 'f_18'], axis = 1)
y = data['target']
#print("\n Feature shape: {}, Target shape: {} \n".format(X.shape, y.shape))



# number of unique values
unique_values = []
for i in data.columns:
    unique_count = data[i].value_counts().count()
    unique_values.append([unique_count])
unique_data = pd.DataFrame(unique_values, index = data.columns, columns = ['Unique values count'] )
#print(unique_data)



# take categorial and numerical data for pipeline usage        
cat_var = ['f_30', 'f_29','f_16','f_15','f_14','f_13','f_11','f_10','f_09','f_08', 'f_27']
num_var = ['f_28','f_26','f_25','f_24','f_23','f_22','f_21','f_20','f_19','f_05', 'f_02','f_01','f_00']
