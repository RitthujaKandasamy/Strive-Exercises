from timeseries2 import data, x, y
from new_features import new_data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2





# plot correlation
data.corr()['T (degC)'].abs().sort_values().plot.barh()
#plt.show()



matrix = new_data.corr()
sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
#plt.show()

matrix = data.corr()
sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()
