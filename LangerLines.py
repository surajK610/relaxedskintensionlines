import os
from flask import Flask, request, redirect, url_for
from flask import flash
from werkzeug.utils import secure_filename
import base64
from line_parts import draw_lines_frontal, draw_lines_frontal_file, draw_lines_side_file
import uuid
import datetime
import json

UPLOAD_FOLDER = '/home/LangerLines/Faces'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

application = Flask(__name__)
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@application.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename, ext = os.path.splitext(filename)
            uid_name = str(uuid.uuid4())
            filename = uid_name + ext
            if '0' in request.form.get('option'):
                path = application.config['UPLOAD_FOLDER']  + "/" +  "Straight" + "/" + datetime.datetime.today().strftime('%Y-%m-%d')
            elif '1' in request.form.get('option'):
                path = application.config['UPLOAD_FOLDER']  + "/" +  "Side" + "/" + datetime.datetime.today().strftime('%Y-%m-%d')

            if not os.path.isdir(path):
                os.mkdir(path)
            file.save(os.path.join(path,filename))

            if '0' in request.form.get('option'):
                draw_lines_frontal_file(os.path.join(path, filename))
            elif '1' in request.form.get('option'):
                draw_lines_side_file(os.path.join(path, filename))
            # return redirect(url_for('uploaded_file',
            #                         filename=filename))
            with open(os.path.join(path, filename), "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
            jsonResp={"processedImage": encoded_string}
	    return json.dumps(jsonResp)
    return '''
    <!doctype html>
    <title>Draw Skin Tension Lines</title>
    <h1>Draw Skin Tension Lines</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
      <input name=option>
      <input type=submit value=Upload>
    </form>
    '''


# import base64
# imgdata = base64.b64decode(imgstring)
# filename = 'some_image.jpg'  #decode base64 string to image
# with open(filename, 'wb') as f:
#     f.write(imgdata)


# '''      <select id=menu name=menu>
#       <option value="0" selected>straight-frontal portrait</option>
#       <option value="1">semi-side portrait</option>
#       </select>
#       '''

from flask import send_from_directory

@application.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(application.config['UPLOAD_FOLDER'] + "/" + datetime.datetime.today().strftime('%Y-%m-%d'), filename)

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
if __name__=="__main__":
     application.run(host='0.0.0.0', port=8000, threaded=True, debug=False)
