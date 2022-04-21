from pipeline import tree_classifiers
from dataset import x, y, x_test
import pandas as pd



# select best modle(accuracy = 83.277)
best_model = tree_classifiers["Skl GBM"]



# Fit and predict best model with all data, x_test
best_model.fit(x, y)
test_pred = best_model.predict(x_test)
#print(test_pred)



# create dataframe for Survived
sub = pd.DataFrame(test_pred, index = x_test.index, columns = ["Survived"])
print(sub)