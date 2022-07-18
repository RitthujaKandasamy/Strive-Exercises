from utils.db import db
from datetime import datetime


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    email = db.Column(db.String(255), nullable=False, unique=True)
    password = db.column(db.String(20))
    #date_created = db.column(db.DateTime, default=datetime.utcnow)


    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"