import streamlit as st
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import streamlit.components.v1 as stc
import torch
import io
from PIL import Image
import numpy as np




# to store the data
st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=True)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 458)
        self.fc2 = nn.Linear(458, 246)
        self.fc3 = nn.Linear(246, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)
        
        # Dropout module with a 0.2 drop probability 
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # Flatten the input tensor
        x = x.view(x.shape[0], -1)    
        # Set the activation functions
        layer1 = self.dropout(F.relu(self.fc1(x)))
        layer2 = self.dropout(F.relu(self.fc2(layer1)))
        layer3 = self.dropout(F.relu(self.fc3(layer2)))
        layer4 = self.dropout(F.relu(self.fc4(layer3)))
        
        out = self.fc5(layer4)
    
        return out

model = Network()

# load the model used only for CNN
def load_model():
    
    model.load_state_dict(torch.load('fashion/model.pth'))

    return model


# load labels from the text
def load_labels():
    labels_path = 'fashion/images_classification.txt'
    # labels_file = os.path.basename(labels_path)
    # if not os.path.exists(labels_file):
    #     wget.download(labels_path)
    with open(labels_path, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories


def predict(model, categories, image):
    preprocess = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))
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
    uploaded_file = st.file_uploader(
        label='Pick an image to test', type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data, caption="Input Image", width=1000)
        return Image.open(io.BytesIO(image_data))

    else:
        st.write('Waiting for upload....')
        return None



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