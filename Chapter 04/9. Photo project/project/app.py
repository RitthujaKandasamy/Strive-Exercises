from io import BytesIO
from flask import Flask, render_template, request, send_file, Response
from werkzeug.utils import secure_filename
from db import db
from user import Upload



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 


db.init_app(app)                 ## initialize the database
db.create_all(app=app)           ## create tables  


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return 'No pic uploaded!', 400

        filename = secure_filename(file.filename)
        if not filename:
             return 'Bad upload!', 400

        upload = Upload(filename=file.filename, data=file.read())
        db.session.add(upload)
        db.session.commit()

        return f"Uploaded: {file.filename}"
    return render_template('index.html')



@app.route('/download/<upload_id>')
def download(upload_id):
    upload = Upload.query.filter_by(id=upload_id).first()
    if not upload:
        return 'Img Not Found!', 404

    return Response( attachment_filename=upload.filename, as_attachment=True)


# @app.route('/<int:id>')
# def get_img(id):
#     img = Upload.query.filter_by(id=id).first()
#     if not img:
#         return 'Img Not Found!', 404

#     return Response(img.img, mimetype=img.mimetype)