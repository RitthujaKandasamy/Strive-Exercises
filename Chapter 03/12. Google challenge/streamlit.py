import streamlit as st
import pandas as pd
from time import sleep
import torch
import numpy as np
import requests
import streamlit.components.v1 as stc
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
#from model import RNN






# app logo
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

with st.sidebar:

    lottie_url = "https://assets8.lottiefiles.com/packages/lf20_c9hh3d5z.json"
    lottie_json = load_lottieurl(lottie_url)
    st_lottie(lottie_json, height=300)




# Menu
with st.sidebar:
    
    app_mode = option_menu(None, ["Home", "App", "Logout"],
                        icons=['house', 'book', 'lock'],
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
    HTML_BANNER = """
    <div style="background-color:#464e5f;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;">Postural-Transitions-RNN</h1>
    </div>
    """
    stc.html(HTML_BANNER)
    
    st.markdown('\n')
    st.markdown('\n')
    st.image("Downloads\\Fitness-Apps.jpg")
    
    
    # Description
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('The main goal is to properly classify human activity based on smartphone sensor, applying the techniques with RNNs.')
    st.markdown('One of the main ideas behind this project is to making it more precise for users.')
    st.markdown("Raw data was given to us in order to train our models and try to predict outcomes for the user.")
    st.markdown('\n')
    st.markdown('\n')

    
    # Team 
    st.title('**Our Team Members**')
    st.subheader(" Kingsley | Ritthuja | Felix ")





# App page
elif app_mode == 'App':
    HTML_BANNER = """
    <div style="background-color:#464e5f;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;">Postural-Transitions-RNN</h1>
    </div>
    """
    stc.html(HTML_BANNER)




    # # RNN Model

    # # load model
    # model = RNN()
    # model.load_state_dict(torch.load(""))


    # # load data
    # data = pd.read_csv('')

    # x = torch.from_numpy(data[:32].values).unsqueeze(0).float()

    # model.eval()
    # with torch.no_grad():
    #     preds = model.forward(x)


    # # Prediction
    # demo = st.radio('Prediction demo', ['start', 'stop'])
        

    # for _, pred_class in data.iterrows():
    #     pred = torch.max(preds, dim=1)
    #     pred = pred_class.item()
    #     if demo == 'start':
    #         placeholder = st.empty()
            

    #         if pred == '1':
    #             st.title("WALKING")
                
    #             sleep(2)
    #             placeholder.empty()


    #         elif pred == '2':
    #             st.title("WALKING_UPSTAIRS")

    #             sleep(2)
    #             placeholder.empty()    

                
    #         elif pred == '3':
    #             st.title("WALKING_DOWNSTAIRS")

    #             sleep(2)
    #             placeholder.empty() 

    #         elif pred == '4':
    #             st.title("SITTING")
                
    #             sleep(2)
    #             placeholder.empty()    
            
    #         elif pred == '5':
    #             st.title("STANDING")
                
    #             sleep(2)
    #             placeholder.empty()    

            
    #         elif pred == '6':
    #             st.title("LAYING ")
                
    #             sleep(2)
    #             placeholder.empty()    
            
    #         elif pred == '7':
    #             st.title("STAND_TO_SIT")
                
    #             sleep(2)
    #             placeholder.empty()    
            
    #         elif pred == '8':
    #             st.title("SIT_TO_STAND")
                
    #             sleep(2)
    #             placeholder.empty()    

    #         elif pred == '9':
    #             st.title("SIT_TO_LIE")
                
    #             sleep(2)
    #             placeholder.empty() 

    #         elif pred == '10':
    #             st.title("LIE_TO_SIT ")

    #             sleep(2)
    #             placeholder.empty()    
            
    #         elif pred == '11':
    #             st.title("STAND_TO_LIE")

    #             sleep(2)
    #             placeholder.empty()    

    #         else:
    #             st.title("LIE_TO_STAND")
                
    #             sleep(2)
    #             placeholder.empty()

                    

    #     else:
    #         break






# thank you 
else:
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()


    lottie_logout = load_lottieurl("https://assets1.lottiefiles.com/private_files/lf30_tapgoijy.json")
    st_lottie(lottie_logout)