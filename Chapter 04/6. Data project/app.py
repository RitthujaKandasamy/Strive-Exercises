from flask import Flask, render_template

## create instance
app = Flask(__name__)


@app.route("/")  ## --> decorator
def index():
    return render_template("home.html")

@app.route("/membership")  ## --> decorator
def membership():
    return render_template("membership.html")

# @app.route("/project")  ## --> decorator
# def project():
#     return render_template("Project.html")

# @app.route("/contact")  ## --> decorator
# def contact():
#     return render_template("Contact.html")

## to run your app
if __name__ =="__main__":
    app.run(debug=True)