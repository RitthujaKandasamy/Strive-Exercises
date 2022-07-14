from flask import Flask
from flask_restful import Api
from routes.home.route import HomeRoute
from utils.db import db



UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}



app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

api = Api(app)

db.init_app(app)                 ## initialize the database
db.create_all(app=app)           ## create tables  

api.add_resource(HomeRoute, '/')


if __name__ =="__main__":
    app.run(debug=True)



