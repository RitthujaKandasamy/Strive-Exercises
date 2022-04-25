# Important libraries
import pandas as pd
from  import train_test_sets, mean_cross_val_score

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
