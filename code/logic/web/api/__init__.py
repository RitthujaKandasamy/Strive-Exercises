from flask import Flask, request
from flask_restful import Api, Resource, reqparse
from flask_sqlalchemy import SQLAlchemy



app = Flask(__name__)
api = Api(app)
app.config.from_object("api.config.Config")
db = SQLAlchemy(app)



class Striver(db.Model):

    __tablename__ = 'strivers'
    id = db.Column(db.Integer, primary_key=True, unique=True)
    email = db.Column(db.String(64), nullable=False, unique=True)
    name = db.Column(db.String(64), nullable=True, unique=False)


    def __init__(self, email, name=""):
            
            self.name = name
            self.email = email
    



class Striver_api(Resource):

    def get(self):
        # parser = reqparse.RequestParser()
        # parser.add_argument("email", type=str)

        # args = parser.parse_args()                                # getting an a argument, in dictionary form ( parser.parse_args()) act as an a key
        email_ = request.args["email"]

        try:
            strive_info = db.session.query(Striver).filter_by(email=email_).first()
            return {"Name": strive_info.name, "Email": strive_info.email}

        except:
            return{"ERROR": "Couldn't find email"}
        
        


    def post(self):
        # parser = reqparse.RequestParser()
        # parser.add_argument("email", type=str)
        # parser.add_argument("name", type=str)

        # args = parser.parse_args()                                # getting an a argument, in dictionary form ( parser.parse_args()) act as an a key
        email_ = request.form["email"]
        name_ = request.form["name"]

        try:
            db.session.add(Striver(email=email_, name=name_))
            db.session.commit()

            return {"email":email_, "name":name_}

        except:
            return{"ERROR": "Couldn't insert email"}




api.add_resource(Striver_api, '/striver')
