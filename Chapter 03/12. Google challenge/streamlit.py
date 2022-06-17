import streamlit as st
import pandas as pd
from time import sleep
import torch
import numpy as np
import streamlit.components.v1 as stc
from streamlit_option_menu import option_menu
from model import RNN







# title
HTML_BANNER = """
<div style="background-color:#464e5f;padding:10px;border-radius:10px">
<h1 style="color:white;text-align:center;"></h1>
</div>
"""
stc.html(HTML_BANNER)




# RNN Model

# load model
model = RNN()
model.load_state_dict(torch.load(""))


# load data
data = pd.read_csv('')

x = torch.from_numpy(data[:32].values).unsqueeze(0).float()

model.eval()
with torch.no_grad():
    preds = model.forward(x)


# Prediction
demo = st.radio('Prediction demo', ['start', 'stop'])
      

for _, pred_class in data.iterrows():
    pred = torch.max(preds, dim=1)
    pred = pred_class.item()
    if demo == 'start':
        placeholder = st.empty()
        

        if pred == '1':
            st.title("WALKING")
            
            sleep(2)
            placeholder.empty()


        elif pred == '2':
            st.title("WALKING_UPSTAIRS")

            sleep(2)
            placeholder.empty()    

            
        elif pred == '3':
            st.title("WALKING_DOWNSTAIRS")

            sleep(2)
            placeholder.empty() 

        elif pred == '4':
            st.title("SITTING")
            
            sleep(2)
            placeholder.empty()    
        
        elif pred == '5':
            st.title("STANDING")
            
            sleep(2)
            placeholder.empty()    

        
        elif pred == '6':
            st.title("LAYING ")
            
            sleep(2)
            placeholder.empty()    
        
        elif pred == '7':
            st.title("STAND_TO_SIT")
            
            sleep(2)
            placeholder.empty()    
        
        elif pred == '8':
            st.title("SIT_TO_STAND")
            
            sleep(2)
            placeholder.empty()    

        elif pred == '9':
            st.title("SIT_TO_LIE")
            
            sleep(2)
            placeholder.empty() 

        elif pred == '10':
            st.title("LIE_TO_SIT ")

            sleep(2)
            placeholder.empty()    
        
        elif pred == '11':
            st.title("STAND_TO_LIE")

            sleep(2)
            placeholder.empty()    

        else:
            st.title("LIE_TO_STAND")
            
            sleep(2)
            placeholder.empty()

                

    else:
        break
