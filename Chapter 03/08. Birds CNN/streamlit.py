import streamlit as st
#import tensorflow as tf
import numpy as np
import torch
from PIL import Image, ImageOps


st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)

def load_model():
	model = torch.load('C:\\Users\\ritth\\code\\Strive\\CNN-Weekend-Challenge\\model.pth')
	return model


def predict_class(img, model):

	# image = tf.cast(image, tf.float32)
	# image = tf.image.resize(image, [180, 180])

	# image = np.expand_dims(image, axis = 0)

	# prediction = model.predict(image)

	# return prediction
    

	# Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 124, 124, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (124, 124)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability



model = load_model()
st.title('Flower Classifier')

file = st.file_uploader("Upload an image of a flower", type=["jpg", "png"])


if file is None:
	st.text('Waiting for upload....')

else:
	slot = st.empty()
	slot.text('Running inference....')

	test_image = Image.open(file)

	st.image(test_image, caption="Input Image", width = 400)

	pred = predict_class(np.asarray(test_image), model)

	class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

	result = class_names[np.argmax(pred)]

	output = 'The image is a ' + result

	slot.text('Done')

	st.success(output)