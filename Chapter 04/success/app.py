#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import torch
import cv2
import mediapipe as mp
import time
from random import random
from english_words import english_words_lower_alpha_set
from flask import Flask, render_template, Response, request
import torch.nn as nn
from torchvision import transforms as transforms
import torch.nn.functional as F


# In[5]:


global freestyle, switch
freestyle=0
switch=1


# In[6]:

train_transform = transforms.Compose([
                                    transforms.Resize((224, 224)),           # resize will resize all the images into same scale(same pixels) given images size are small and big, so we take approximitily 50
                                    transforms.RandomRotation(20),
                                    transforms.RandomResizedCrop(200),      # crop adjust the images to other images in features and we taken it as 28 smaller than resize
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],        # normalize have mean and standard deviation for color pic (red, green, blue)
                                        std=[0.229, 0.224, 0.225]) 
                                    ])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 16, 5)
        self.fc1 = nn.Linear(16*47*47, 220)
        self.fc2 = nn.Linear(220, 184)
        self.fc3 = nn.Linear(184, 93)
        self.fc4 = nn.Linear(93, 3)

        # Dropout module with a 0.2 drop probability 
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        layer1 = self.dropout(F.relu(self.fc1(x)))
        layer2 = self.dropout(F.relu(self.fc2(layer1)))
        layer3 = self.dropout(F.relu(self.fc3(layer2)))
        out = F.log_softmax(self.fc4(layer3), dim=1)
    
        return out

model = Net()

clf = model.load_state_dict(torch.load('C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 04\\success\\model1.pth'))



# In[4]:


app = Flask(__name__, template_folder='./templates')


# In[43]:


def camera_max():
    '''Returns int value of available camera devices connected to the host device'''
    camera = 0
    while True:
        if (cv2.VideoCapture(camera).grab()):
            camera = camera + 1
        else:
            cv2.destroyAllWindows()
            return(max(0,int(camera-1)))
        
cam_max = camera_max()


# In[48]:


cap = cv2.VideoCapture(cam_max, cv2.CAP_DSHOW)


# In[ ]:


letters = ['A', 'B', 'C']
words = [i for i in sorted(list(english_words_lower_alpha_set)) if 'z' not in i and len(i) > 3 and len(i) <= 10]
start_time = time.time()
curr_time = 0
easy_word_user = ''
eraser = 0
easy_word = words[int(random()*len(words))].upper()
easy_word_index = 0



# In[ ]:


def easy_mode(frame):
    global cap, curr_time
    
    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        results = model.process(image)                 # Make prediction
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
        return image, results

    def get_landmark_dist_test(results, x, y):
        hand_array = []
        wrist_pos = results.multi_hand_landmarks[0].landmark[0]
        for result in results.multi_hand_landmarks[0].landmark:
            hand_array.append((result.x-wrist_pos.x) * (width/x))
            hand_array.append((result.y-wrist_pos.y) * (height/y))
        return(hand_array[2:])


    #Main function
    #cap = cv2.VideoCapture(cam_max)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set mediapipe model
    mp_hands = mp.solutions.hands # Hands model
    with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1) as hands:
        while cap.isOpened():

            # Read feed
            #ret, frame = cap.read()

            try:
                cv2.putText(frame, easy_word, (int(width*0.05), int(height*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_4)
                cv2.putText(frame, easy_word_user, (int(width*0.05), int(height*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_4)
            except Exception as e:
                print(e)

            # Make detections
            image, results = mediapipe_detection(frame, hands)

            #letter_help = cv2.resize(cv2.imread('easy_model_letter/{}.png'.format(easy_word[easy_word_index].lower())), (0,0), fx=0.2, fy=0.2)

            #Find bounding box of hand
            if results.multi_hand_landmarks:
                x = [None,None]
                y=[None,None]
                for result in results.multi_hand_landmarks[0].landmark:
                    if x[0] is None or result.x < x[0]: x[0] = result.x
                    if x[1] is None or result.x > x[1]: x[1] = result.x

                    if y[0] is None or result.y < y[0]: y[0] = result.y
                    if y[1] is None or result.y > y[1]: y[1] = result.y


                if curr_time < round((time.time() - start_time)/3,1) and x[0] is not None:
                        curr_time = round((time.time() - start_time)/3,1)
                        try:
                            test_image = get_landmark_dist_test(results, x[1]-x[0], y[1]-y[0])
                            test_pred = np.argmax(clf.predict_proba(np.array([test_image])))
                            test_probs = clf.predict_proba(np.array([test_image]))[0]
                            print("Predicted:",letters[test_pred], ", pred prob:", max(test_probs), ", current index:", easy_word_index, ", current time:", curr_time)
                            if max(test_probs) >= 0.8 or (max(test_probs) >= 0.6 and letters[test_pred] in ['A', 'B', 'C']):
                                pred_letter = letters[test_pred].upper()
                                if easy_word_index < len(easy_word) and pred_letter == easy_word[easy_word_index] and (easy_word_index == 0 or easy_word[easy_word_index] != easy_word[easy_word_index - 1]):
                                    easy_word_user += pred_letter
                                    easy_word_index += 1
                                    location = results.multi_hand_landmarks[0].landmark[0].x
                                if easy_word_index < len(easy_word) and pred_letter == easy_word[easy_word_index] and easy_word_index > 0 and easy_word[easy_word_index] == easy_word[easy_word_index - 1] and abs(location - results.multi_hand_landmarks[0].landmark[0].x) > 0.1:
                                    easy_word_user += pred_letter
                                    easy_word_index += 1
                                    location = results.multi_hand_landmarks[0].landmark[0].x

                            if easy_word_user == easy_word:
                                time.sleep(0.5)
                                easy_word = words[int(random()*len(words))].upper()
                                easy_word_index = 0
                                easy_word_user = ''

                        except Exception as e:
                            print(e)

            # Show letter helper
            #frame[5:5+letter_help.shape[0],width-5-letter_help.shape[1]:width-5] = letter_help

            return frame
            
    return frame




# In[ ]:


def sign_frame():  # generate frame by frame from camera
    global easy, cap
    while True:
        success, frame = cap.read() 
        if success:
            if(easy):                
                frame = easy_mode(frame)
            
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass


# In[ ]:


@app.route('/')
def index():
    return render_template("index.html")


# In[ ]:


@app.route('/video_feed')
def video_feed():
    return Response(sign_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


# In[ ]:


@app.route('/requests',methods=['POST','GET'])
def mode():
    global switch, free
    if request.method == 'POST':
        
        if  request.form.get('free') == 'Freestyle':
            free=not free  
            
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


