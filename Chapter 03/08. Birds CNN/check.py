import streamlit as st
import torch
import os
import wget
import io
from PIL import Image
import numpy as np
from torchvision import transforms, models




st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)





def load_model():
    model = models.resnext50_32x4d(pretrained = True)
    inputs = model.fc.in_features
    outputs = 6

    model.fc = torch.nn.Linear(inputs, outputs) 
    model.load_state_dict( torch.load('C:\\Users\\ritth\\code\\Strive\\CNN-Weekend-Challenge\\model.pth') )
    
    return model



def load_labels():
    labels_path = 'C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 03\\08. Birds CNN\\images_classification.txt'
    # labels_file = os.path.basename(labels_path)
    # if not os.path.exists(labels_file):
    #     wget.download(labels_path)
    with open(labels_path, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories



def predict(model, categories, image):
    preprocess = transforms.Compose([
                                        transforms.Resize(150),
                                        transforms.CenterCrop(124),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225]),
                                    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100
    st.header("The image is a " + categories[np.argmax(probabilities)])

    
    top5_prob, top5_catid = torch.topk(probabilities, 6)
    with st.expander("Calculating results... "):
         for i in range(top5_prob.size(0)):
            st.write(categories[top5_catid[i]], top5_prob[i].item())
    



def load_image():
    uploaded_file = st.file_uploader(label = 'Pick an image to test', type=["jpg", "png"])
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data, caption="Input Image", width = 400)
        return Image.open(io.BytesIO(image_data))
    
    else:
        st.write('Waiting for upload....')
        return None




def main():
    st.title('Intel Image Classification Application')
    model = load_model()
    categories = load_labels()
    image = load_image()  
    result = st.button('Run on image')
    if result:
             predict(model, categories, image)
    


if __name__ == '__main__':
    main()