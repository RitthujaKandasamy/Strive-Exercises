from flask import Flask,render_template

## create instance
app = Flask(__name__)


@app.route("/")  ## --> decorator
def index():
    return render_template("home.html")



## to run your app
if __name__ =="__main__":
    app.run(debug=True)