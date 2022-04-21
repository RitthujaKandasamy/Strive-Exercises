import pandas   as pd



# read dataset both train and test for Titanic from Kaggle

df      = pd.read_csv("C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 02\\08. Robust ML\\titanic\\train.csv", index_col = 'PassengerId')
df_test = pd.read_csv("C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 02\\08. Robust ML\\titanic\\test.csv",  index_col = 'PassengerId')

print("Train DataFrame:", df.shape)
print("Test DataFrame: ", df_test.shape)


# clean data 
# remove name and surname by split('.')[0].split(',')[1]
# get only Title like Mr, Mrs......

df['Title'] = df['Name'].apply(lambda df: df.split('.')[0].split(',')[1].strip())
df_test['Title'] = df_test['Name'].apply(lambda df_test: df_test.split('.')[0].split(',')[1].strip())


# create dict. to get better result

title_dictionary = {
                        "Capt": "Officer",
                        "Col": "Officer",
                        "Major": "Officer",
                        "Jonkheer": "Royalty",
                        "Don": "Royalty",
                        "Sir" : "Royalty",
                        "Dr": "Officer",
                        "Rev": "Officer",
                        "the Countess":"Royalty",
                        "Mme": "Mrs",
                        "Mlle": "Miss",
                        "Ms": "Mrs",
                        "Mr" : "Mr",
                        "Mrs" : "Mrs",
                        "Miss" : "Miss",
                        "Master" : "Master",
                        "Lady" : "Royalty"
                                                }


# Use map to apply the title dictionary

df["Title"] =  df['Title'].map(title_dictionary)
df_test["Title"] = df_test['Title'].map(title_dictionary)  



# preprocessing
# ["Survived", 'Name', 'Ticket', 'Cabin'] we droped this columns because
# Survived is Target, 
# 'Name', 'Ticket', 'Cabin' this are not giving related information or less information

x = df.drop(columns = ["Survived", 'Name', 'Ticket', 'Cabin']) 
y = df["Survived"] 


# in Titanic.test we also droped some unwanted columns

x_test = df_test.drop(columns = ['Name', 'Ticket', 'Cabin']) 


# take categorial and numerical data for pipeline usage

cat_vars  = ['Sex', 'Embarked', 'Title']         
num_vars  = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Age']