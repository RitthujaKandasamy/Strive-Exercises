from flask import Flask, redirect, render_template, url_for, flash
from forms import RegistrationForm,LoginForm
from utils.db import db
from utils.models.user import User


## create instance
app = Flask(__name__)
app.config['SECRET_KEY'] = 'thisisfirstflaskapp'
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


db.init_app(app)                 ## initialize the database
db.create_all(app=app)           ## create tables  



@app.route("/")  ## --> decorator
def homepage():
    return render_template("homepage.html", title='Home')


@app.route("/talk")  ## --> decorator
def talk():
    return render_template("talk.html")


@app.route("/sign", methods=['POST', 'GET']) 
def sign():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email= form.email.data).first()
        if form.email.data==User.email and form.password.data==User.password:
            flash(f'Login successful for {form.email.data}', category='success')
            return redirect(url_for('homepage'))
        else:
            flash(f'Login unsuccessful for {form.email.data}', category='danger')
    return render_template("sign.html", form=form)


@app.route("/share", methods=['POST', 'GET'])
def share():
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data, password=form.password.data)
        db.session.add(user)
        db.session.commit()
        flash(f'Account created successfully for {form.username.data}', category='success')
        return redirect(url_for('homepage'))
    return render_template("register.html", form=form)


