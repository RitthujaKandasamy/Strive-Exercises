import joblib
import pandas as pd
import time
import argparse



""" 
HOW TO USE:
1. Open your terminal and set the working directory where the
cli.py file is located.
Example:
cd /d E:\strive_school\work\CHALLENGE\Chap2\Hipocratia

2. Using python command, run the cli.py file followed
csv file where the records are stored.
Example:
python cli.py device/test_data.csv


NOTE: csv file must followed the same structure as `test_data.csv`.
It's the same structure as `X_test`
"""


# Loading model
model = joblib.load('best_model.joblib')



# Setting up the CLI
parser = argparse.ArgumentParser(description='Predict heart attack')

parser.add_argument('file_path', type = str,
                    help ='Path to file where user records are stored')

args = parser.parse_args()

path_to_records = args.file_path
# print(args)



# Loading records
user_data = pd.read_csv(path_to_records)
# column_names = user_data.columns




# Predictions
for _, row in user_data.iterrows():
    x = pd.DataFrame([row])

    print(x.to_string(index=False), '\n')

    pred = model.predict(x)[0]
    # print(pred)
    if pred == 1:
        print("You are prone to heart attack...Please take your drugs!!!")
    else:
        print("No risk of heart attack")
    print('-'*100, '\n\n')

    # Pause (to simulate readings every 3 secondes)
    time.sleep(3)