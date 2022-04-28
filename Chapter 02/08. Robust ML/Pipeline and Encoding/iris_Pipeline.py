from iris_dataset import num_vars
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



# pipeline for categorical and numerical data
num_preprocessing = Pipeline( [('imp', SimpleImputer(fill_value= -999, strategy='constant')) ] )
#cat_preporcessing = Pipeline( [('ordinal', OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value = 10)),
                               # ('imp', SimpleImputer(strategy='constant', fill_value='missing'))] )




tree_prepr = ColumnTransformer( [('num', num_preprocessing, num_vars)], remainder = 'passthrough') 

print(tree_prepr)




# create dict. to store all tree
tree_classifiers = {
                      "Decision Tree": DecisionTreeClassifier(random_state=0),
                      "Extra Trees": ExtraTreesClassifier(random_state=0),
                      "Random Forest": RandomForestClassifier(random_state=0),
                      "AdaBoost": AdaBoostClassifier(random_state=0),
                      "Skl GBM": GradientBoostingClassifier(random_state=0),
                      "Skl HistGBM": HistGradientBoostingClassifier(random_state=0),
                      "XGBoost": XGBClassifier(),
                      "LightGBM": LGBMClassifier(random_state=0),
                      "CatBoost": CatBoostClassifier(random_state=0)
                      }
                      


tree_classifiers = {name: make_pipeline(tree_prepr, model) for name, model in tree_classifiers.items()}

