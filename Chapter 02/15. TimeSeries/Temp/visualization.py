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



# apply SelectKBest class to extract top best features
bestfeatures = SelectKBest(score_func = chi2, k = 10)
fit = bestfeatures.fit(new_data, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(new_data.columns) 
featureScores = pd.concat([dfcolumns, dfscores], axis = 1)
featureScores.columns = ['Specs', 'Score']  
print(featureScores.nlargest(15, 'Score'))  
