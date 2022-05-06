from timeseries2 import data, x, y
from new_features import new_data
import matplotlib.pyplot as plt
import seaborn as sns





# plot correlation
data.corr()['T (degC)'].abs().sort_values().plot.barh()
#plt.show()


# matrix new
matrix = new_data.corr()
sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
#plt.show()



for col_name in new_data.columns:
        plt.figure()
        plt.hist(new_data[col_name])
        plt.title(col_name)
        plt.show()