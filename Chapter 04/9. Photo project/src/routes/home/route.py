from flask_restful import Resource
import os
from app import ALLOWED_EXTENSIONS, app
from flask import request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from flask import send_from_directory



class HomeRoute(Resource):
    def allowed_file(filename):
        return '.' in filename and \
             filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


    @app.route('/', methods=['GET', 'POST'])
    def upload_file():
        if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file:
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                return redirect(url_for('download_file', name=filename))
        return '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data>
        <input type=file name=file>
        <input type=submit value=Upload>
        </form>
        '''

    @app.route("/uploads/<path:name>") # <--- name is file name
    def download_file(name):
		
        return send_from_directory(
               app.config['UPLOAD_FOLDER'], name, as_attachment=True
    )



