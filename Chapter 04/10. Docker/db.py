from sqlalchemy import create_engine
import pandas as pd


def create_db():
    
    try:
        engine = create_engine("sqlite:///beer.db")     # create a new db
        conn = engine.connect()                         # if there is already db exist don't create, if not create one

        df = pd.read_csv("beers.csv")
        print(df.head())

        df.to_sql("beer", conn, if_exists='replace')
        conn.close()

        return 0

    except:
        return 1

def get_beers():

    try:
        engine = create_engine("sqlite:///beer.db")  
        conn = engine.connect()                         
        
        df = pd.read_sql_query("SELECT * FROM beers", conn)
        
        conn.close()

        return df.to_dict()                                  # return df in dictionary form, because Api accept Json

    except:
        return {"ERROR": "No table was found"}


def get_origin(Origin):

    try:
        engine = create_engine("sqlite:///beer.db")     
        conn = engine.connect()                         

        df = pd.read_sql_query("SELECT Name FROM beers WHERE Origin = '{}'".format(Origin), conn)
        conn.close()

        return df.to_dict()                                 

    except:
        return {"ERROR": "No origin was found"}


create_db()