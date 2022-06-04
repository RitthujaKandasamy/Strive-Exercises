from PIL import Image
from torchvision import models, transforms
import torch
import io
import streamlit as st



# set title of app
st.title("Intel Image Classification Application")
st.write("")



# enable users to upload images for the model to make predictions
file_up = st.file_uploader(label = 'Pick an image to test', type = ["jpg", "png"])



def predict(image):
    
    # create a ResNet model
    model = models.resnext50_32x4d(pretrained = True)
    inputs = model.fc.in_features
    outputs = 6

    model.fc = torch.nn.Linear(inputs, outputs) 
    model.load_state_dict( torch.load('C:\\Users\\ritth\\code\\Strive\\CNN-Weekend-Challenge\\model.pth') )
    

    # transform the input image through resizing, normalization
    transform = transforms.Compose([
                                    transforms.Resize(150),
                                    transforms.CenterCrop(124),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean = [0.485, 0.456, 0.406],
                                        std = [0.229, 0.224, 0.225]
                                    )])



    # load the image, pre-process it, and make predictions
    img = Image.open(image)
    batch_t = torch.unsqueeze(transform(img), 0)
    model.eval()
    output = model(batch_t)

    with open('C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 03\\08. Birds CNN\\images_classification.txt') as f:
        classes = [line.strip() for line in f.readlines()]


    # return the top 5 predictions ranked by highest probabilities
    probabilities = torch.nn.functional.softmax(output, dim = 1)[0] * 100
    _, indices = torch.sort(output, descending = True)
    return [(classes[idx], probabilities[idx].item()) for idx in indices[0][:5]]


if file_up is not None:
    # display image that user uploaded
    image_data = file_up.getvalue()
    image = Image.open(io.BytesIO(image_data))
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    labels = predict(file_up)
    

    # print out the top 5 prediction labels with scores
    for i in labels:
        st.write("Prediction: ", i[0], ",   Score: ", i[1])


else:
        st.write('Waiting for upload....')




