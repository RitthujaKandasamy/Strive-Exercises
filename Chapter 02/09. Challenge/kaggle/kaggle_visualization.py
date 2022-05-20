from kaggle_dataset import data
import matplotlib.pyplot as plt
import seaborn as sns



# full dataset to know null value
plt.figure(figsize = (18, 10))
sns.heatmap(data.isnull(), yticklabels = False, cbar = False, cmap = 'viridis' )
plt.title("Null Values in Dataset")
plt.show()


# correlation matrix
plt.figure(figsize = (18, 10))
data.corr()["target"].abs().sort_values().plot.barh()
plt.title("Correlation of features in the Dataset", fontsize = 25)
plt.xlabel("Correlation", fontsize = 20)
plt.ylabel("Features", fontsize = 20)
plt.show()


# target variable distribution
sns.set(rc={"figure.figsize":(8, 6)})
sns.countplot(x ='target', data = data,hue="target")
plt.title("Distribution of the target variable", fontsize = 20)
plt.show()




