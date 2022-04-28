from iris_dataset import irisdata, num_vars
import matplotlib.pyplot as plt
import seaborn as sns



# full dataset to know null value
plt.figure(figsize = (10, 8))
sns.heatmap(irisdata.isnull(), yticklabels = False, cbar = False, cmap = 'viridis' )
plt.title("Null Values in Dataset")
plt.show()


# pairplot to get better view
sns.pairplot(irisdata, hue = 'Class', height = 2.5)
plt.show()


# target variable distribution
sns.countplot(x = irisdata['Class'])
plt.title("Distribution of the target variable")
plt.show()


# categorical values in visualization
for i in num_vars:
    plt.figure(figsize = (8, 6))
    sns.distplot(irisdata[i], hist_kws = dict(linewidth = 1, edgecolor = "k"), bins = 20)
    
    plt.title(i)
    plt.xlabel(i)

    plt.tight_layout()
    plt.show()

