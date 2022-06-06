import streamlit as st
import torch
import os
import wget
import io
from PIL import Image
import numpy as np
import base64
import shutil
from torchvision import transforms, models
import streamlit.components.v1 as stc
from streamlit_option_menu import option_menu






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
        st.image(image_data, caption = "Input Image", width = 1000)
        return Image.open(io.BytesIO(image_data))
    
    else:
        st.write('Waiting for upload....')
        return None






# Menu   
app_mode = option_menu(menu_title = None, 
                    options = ["HOME", "APP", "Classification App", "ABOUT", "LOGOUT"],
                    icons = ['house', 'book', 'app', 'person-circle', 'lock'],
                    menu_icon = "app-indicator", 
                    default_index = 0,
                    orientation = "horizontal",
                    styles = {
    "container": {"padding": "5!important", "background-color": "#f0f2f6"},
    "icon": {"color": "orange", "font-size": "28px"}, 
    "nav-link": {"font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
    "nav-link-selected": {"background-color": "#2C3845"}
})




# Home page
if app_mode == 'HOME':

    # title
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')
   

    HTML_BANNER = """
    <center><div style="background-color:#3a7ff0;padding:10px;border-radius:10px;width:800px">
    <h1 style="color:white;text-align:center;">Natural Scenes Images Classification </h1>
    </div></center>
    """
    stc.html(HTML_BANNER)

    st.markdown('###')
    
    
    # Gif from local file
    file_ = open("Downloads\\img_gif.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<center><img src="data:image/gif;base64,{data_url}" alt="test gif"></center>',
        unsafe_allow_html= True
    )


    # Description
    st.markdown('###')
    st.header("About this app    -->")
    st.markdown('###')
    st.markdown('The Intel Image Classification contains 25k images of size 150x150 distributed under 6 categories: buildings, forest, glacier, mountain, sea, street.')
    st.markdown('The Train, Test and Prediction data is separated in each zip files. There are around 14k images in Train, 3k in Test and 7k in Prediction.')
    st.markdown("This project is about creating a CNN model able to classify an image. Additionally, the app can receive a folder of mixed images, creates subfolders into which to classify the images.")
    st.markdown("...")
    st.markdown('\n')
    st.markdown('\n')
    st.subheader("Updated on")
    st.markdown("June 06, 2022")
    st.markdown('\n')
    st.markdown('\n')
    st.subheader("Data Collected")

    

    

    


# app
elif app_mode == 'APP':
    
    # title
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')

    HTML_BANNER = """
    <center><div style="background-color:#3a7ff0;padding:10px;border-radius:10px;width:800px">
    <h1 style="color:white;text-align:center;">Natural Scenes Images Classification </h1>
    </div></center>
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




# classification app
elif app_mode == 'Classification App':
    
    # title
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')

    HTML_BANNER = """
    <center><div style="background-color:#3a7ff0;padding:10px;border-radius:10px;width:800px">
    <h1 style="color:white;text-align:center;">Natural Scenes Images Classification </h1>
    </div></center>
    """
    stc.html(HTML_BANNER)
    
   
    origin_folder = 'C:\\Users\\ritth\\code\\Strive\\CNN-Weekend-Challenge\\Test_images'
    destination_folder = 'üimages'
    os.mkdir(destination_folder)

    

    # get image file names
    image_names = os.listdir(origin_folder)
    # print(image_names)
    classes = torch.load('C:\\Users\\ritth\\code\\Strive\\CNN-Weekend-Challenge\\classes.pth')
    model = load_model()

    for name in image_names:
        # get prediction
        img_src_pth = os.path.join(origin_folder, name)
        img = Image.open(img_src_pth)
        pred = predict(model,img,classes)

        # Destination subfolder (based on pred) and image pth names
        dst_sub_folder = os.path.join(destination_folder, pred)
        img_dst_pth = os.path.join(dst_sub_folder, name)

        # If subfolder doesn't exist, create it and copy in image
        if not os.path.isdir(dst_sub_folder):
            os.mkdir(dst_sub_folder)
            shutil.copy(img_src_pth, img_dst_pth)

        else:
            shutil.copy(img_src_pth, img_dst_pth)




# about us
elif app_mode == "ABOUT":

   # title
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')

    HTML_BANNER = """
    <center><div style="background-color:#3a7ff0;padding:10px;border-radius:10px;width:800px">
    <h1 style="color:white;text-align:center;"> Our Team Members </h1>
    </div></center>
    """
    stc.html(HTML_BANNER)


    st.image("Downloads\\team.PNG", use_column_width = True)





# logout
else:

    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')

    # Gif from local file
    file_ = open("Downloads\\logout.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<center><img src="data:image/gif;base64,{data_url}" alt="test gif"></center>',
        unsafe_allow_html= True
    )
