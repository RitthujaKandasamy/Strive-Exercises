from flask import Flask
from flask_restful import Api, Resource, reqparse
import db



app = Flask(__name__)
api = Api(app)

class Beers(Resource):

    def get(self):
        return db.get_beers()


class Name(Resource):

    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument("Origin", type=str)

        args = parser.parse_args()
        origin = args["Origin"]
        
        print(origin)

        return db.get_origin(origin)
        #return {"Origin":origin}



api.add_resource(Beers, "/beers")
api.add_resource(Name, "/name")


if __name__ =="__main__":
    app.run('0.0.0.0')
