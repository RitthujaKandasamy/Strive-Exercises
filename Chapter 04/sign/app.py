
from cv2 import threshold
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
from utils import *
from torchvision import transforms, models




app = Flask(__name__)



def generate_video():

    cap = cv2.VideoCapture(0)

    # Hand detector
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.75)

    # Hand landmarks drawing
    mp_drawing = mp.solutions.drawing_utils

    

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(200),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    ])

    #image = transform(image).unsqueeze(0)

    # Load model
    model = models.resnext50_32x4d(pretrained = True)
    model.load_state_dict(torch.load('sign\\model.pth'))

    # Load classes for predictions
    classes = ['A', 'B', 'C']

    # Load image
    #img = Image.open('asl_alphabet_test\\asl_alphabet_test\\C_test.jpg')


    print(predict(classes, model, threshold=0.75))

    mode = 0  # mode normal by default

    while True:

        key = cv2.waitKey(10)
        if key == ord('q'):
            break

        # reset mode and class id
        class_id, mode = select_mode(key, mode)

        # read camera
        has_frame, frame = cap.read()
        if not has_frame:
            break

        # horizontal flip and color conversion for mediapipe
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # landmarks detection
        results = hands.process(frame_rgb)

        # if detection
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # draw landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # get landmarks coordinates
                coordinates_list = calc_landmark_coordinates(frame, hand_landmarks)

                # Conversion to relative coordinates and normalized coordinates
                preprocessed = pre_process_landmark(
                    coordinates_list)

                # Write to the dataset file (if mode == 1)
                logging_csv(class_id, mode, preprocessed)

        frame = draw_info(frame, mode, class_id)
        # cv.imshow('', frame)
        _ , buffer = cv2.imencode('.jpg', frame)
        frame  = buffer.tobytes()
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



    cap.release()
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
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/info")  ## --> decorator
def info():
    return render_template("contact.html")

    

## to run your app
if __name__ =="__main__":
    app.run(debug=True)