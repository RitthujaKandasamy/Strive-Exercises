import streamlit as st
import torch
from torchvision import transforms
import io
from PIL import Image
import numpy as np







st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)


def load_model():
  model = torch.load('C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 03\\05. MLP Kaggle challenge\\checkpoint.pth')
  return model
   
with st.spinner('Model is being loaded..'):
  model = load_model()


st.write("""
         # Dress Classification
         """
         )



file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


import cv2
from PIL import Image, ImageOps
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image_data, model):
    
        size = (28, 28)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = torch.nn.Softmax(predictions[0])
    class_names = ['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot']
    st.write(predictions)
    st.write(score)

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)




# def load_image():
#     uploaded_file = st.file_uploader(label='Pick an image to test')
#     if uploaded_file is not None:
#         image_data = uploaded_file.getvalue()
#         st.image(image_data)
#         return Image.open(io.BytesIO(image_data))
#     else:
#         return None


# def load_labels():
#       categories = ['T-shirt/top',
#                            'Trouser',
#                             'Pullover',
#                             'Dress',
#                             'Coat',
#                             'Sandal',
#                            'Shirt',
#                            'Sneaker',
#                            'Bag',
#                             'Ankle Boot']
#       return categories



# def predict(model, categories, image):
#     preprocess = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     input_tensor = preprocess(image)
#     input_batch = input_tensor.unsqueeze(1)

#     with torch.no_grad():
#         output = model[input_batch]

#     probabilities = torch.nn.functional.softmax(output[0], dim=0)

#     top5_prob, top5_catid = torch.topk(probabilities, 5)
#     for i in range(top5_prob.size(0)):
#         st.write(categories[top5_catid[i]], top5_prob[i].item())

    


# def main():
#     st.title('Pretrained model demo')
#     model = load_model()
#     categories = load_labels()
#     image = load_image()
#     result = st.button('Run on image')
#     if result:
#         st.write('Calculating results...')
#         predict(model, categories, image)



# if __name__ == '__main__':
#     main()



