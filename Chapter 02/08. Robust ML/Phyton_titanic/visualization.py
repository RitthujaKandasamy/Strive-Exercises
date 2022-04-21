from dataset import df_test, df
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Survived to check target
sns.countplot(x = df['Survived'])
plt.title("Train data Survived")
plt.show()


# Survived and Sex used to compare the target
sns.countplot(x ='Survived', hue ='Sex', data = df)
plt.title("Train data Survived and Sex")
plt.show()


# better understand compare age, sex, survived
grid = sns.FacetGrid(df, row = "Sex", col = "Survived", margin_titles = True)
grid.map(plt.hist, "Age", bins = np.linspace(0, 40, 15))
plt.show()


# histogram for age
sns.histplot(x = df['Age'])
plt.title("Train data Age")
plt.show()


# full dataset to know null value
plt.figure(figsize = (10, 8))
sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis' )
plt.title("Full Train data")
plt.show()


# pairplot to get better view
sns.pairplot(df, hue = 'Survived', height = 2.5)
plt.title("Full Train data")
plt.show()


# histogram for age
sns.histplot(x = df_test['Age'])
plt.title("Test data Age")
plt.show()


# full dataset to know null value
plt.figure(figsize = (10, 8))
sns.heatmap(df_test.isnull(), yticklabels = False, cbar = False, cmap = "cubehelix" )
plt.title("Full Test data")
plt.show()
