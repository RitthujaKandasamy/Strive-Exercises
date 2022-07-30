
from flask import Flask, render_template, Response
import cv2




app = Flask(__name__)


def generate_frames():
    
    camera=cv2.VideoCapture(0)

    while True:
        # key = cv2.waitKey(10)
        # if key == ord('q'):
        #     break

        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    camera.release()
    cv2.destroyAllWindows()



@app.route("/")  ## --> decorator
def index():
    return render_template("home.html")

@app.route("/about")  ## --> decorator
def about():
    return render_template("about.html")

@app.route("/pro")  ## --> decorator
def project():
    return render_template("project.html")

@app.route("/cam")  ## --> decorator
def cam():
    return render_template("camera.html")

@app.route("/video")  ## --> decorator
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/info")  ## --> decorator
def info():
    return render_template("contact.html")

    

## to run your app
if __name__ =="__main__":
    app.run(debug=True)