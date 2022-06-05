import streamlit as st
import torch
import os
import wget
import io
from PIL import Image
import numpy as np
import requests
import base64
from torchvision import transforms, models
import streamlit.components.v1 as stc
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie





# to store the data
st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)




# load the model used only for CNN
def load_model():
    model = models.resnext50_32x4d(pretrained = True)
    inputs = model.fc.in_features
    outputs = 6

    model.fc = torch.nn.Linear(inputs, outputs) 
    model.load_state_dict( torch.load('C:\\Users\\ritth\\code\\Strive\\CNN-Weekend-Challenge\\model.pth') )
    
    return model



# load labels from the text
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


    # pred top 5 labels from the text
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




# app logo
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

with st.sidebar:

    lottie_url = "https://assets1.lottiefiles.com/packages/lf20_0zv8teye.json"
    lottie_json = load_lottieurl(lottie_url)
    st_lottie(lottie_json, height=300)







# Menu

with st.sidebar:
    
    app_mode = option_menu(None, ["Home", "App"],
                        icons=['house', 'person-circle'],
                        menu_icon="app-indicator", default_index=0,
                        styles={
        "container": {"padding": "5!important", "background-color": "#f0f2f6"},
        "icon": {"color": "orange", "font-size": "28px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#2C3845"}
    }
    )





# Home page
if app_mode == 'Home':

    # title
    HTML_BANNER = """
    <div style="background-color:#161c82;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;">Intel Image Classification Application </h1>
    </div>
    """
    stc.html(HTML_BANNER)
    


    # Gif from local file
    file_ = open("Downloads\\travel_2.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="test gif">',
        unsafe_allow_html= True
    )


    # Description
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('The main goal is to develop a fitness and user transport mode detection software able to be used in plug and play style into most apps and smart watches.')
    st.markdown('One of the main ideas behind the project is to facilitate the transport mode detection and calories counting, making it more precise.')
    st.markdown("Raw data was given to us in order to train our ML models and try to predict outcomes for the user.")
    
    
    # Team Img
    st.title('**Our Team**')
    st.image("Downloads\\team.PNG", use_column_width = True)


    # Plot for learning curve
    st.title('**Our result**')
    st.subheader('**Learning Curve**')
    st.image("Downloads\\learning_curve.PNG", use_column_width = True)
    with st.expander('See explanation'):
         st.write('The white part on the plot represent the missing values.')








# Sign in
elif app_mode == 'App':
    

    # title
    HTML_BANNER = """
    <div style="background-color:#161c82;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;">Intel Image Classification Application </h1>
    </div>
    """
    stc.html(HTML_BANNER)

    def main():
        
        model = load_model()
        categories = load_labels()
        image = load_image()  
        result = st.button('Run on image')

        if result:
                 predict(model, categories, image)



    if __name__ == '__main__':
             main()