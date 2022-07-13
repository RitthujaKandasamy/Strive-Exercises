from flask_restful import Resource
from flask import request
import uuid
from utils.models.user import User
from utils.db import db


data = []

class HomeRoute(Resource):
    def get(self):
        users = db.session.query(User).all()
        users = [user.to_json() for user in users]
        return {'data': users}


    def post(self):
        # id = str(uuid.uuid4())
        id=str(uuid.uuid4())
        name = request.form["first_name"]
        last_name = request.form["last_name"]
        email = request.form["email"]
        # user = {'id':id, 'name': name, 'last_name':last_name, 'email':email}
        user = User(user_id=id, first_name= name, last_name=last_name, email=email)
        db.session.add(user)           ## add user to the database
        db.session.commit()            ## commit the changes to the database
        # data.append(user)
        return {'data': user.to_json()}

    
# def find_object_by_id(id):
#     for data_object in data:
#         if data_object["id"] == id:
#             return data_object
#         else:
#             return None


class HomeRouteWithId(Resource):
    def get(self,id):
        # data_object = find_object_by_id(id)
        data_object = db.session.query(User).filter(User.user_id == id).first()
        if(data_object):
            return {"data":data_object.to_json()}
        else:
            return {"data":"Not Found"},404

    
    def put(self,id):
        data_object = db.session.query(User).filter(User.user_id == id).first()
        #data_object = find_object_by_id(id)
        if(data_object):
            for key in request.form.keys():
                setattr(data_object,key,request.form[key])
            db.session.commit() 
            # data_object["name"] = request.form["name"]
            # data_object["last_name"] = request.form["last_name"]
            # data_object["email"] = request.form["email"]
            return {"data":data_object.to_json()}
        else:
            return {"data":"Not Found"},404
    

    def delete(self,id):
        #data_object = find_object_by_id(id)
        data_object = db.session.query(User).filter(User.user_id == id).first()
        if(data_object):
            db.session.delete(data_object)          
            db.session.commit() 
            # data.remove(data_object)
            return {"data":"DELETED"}
        else:
            return {"data":"Not Found"},404

