import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor10Epochs.h5')

image = cv2.imread('C:\\Users\\lenovo\\Documents\\MLProjects\\Brain_Tumor\\predict\\pred0.jpg')
img = Image.fromarray(image)
img = img.resize((64,64))
img = np.array(img)
# print(img)

input_img = np.expand_dims(img, axis=0)

# #for binary model
# # result = (model.predict(input_img) > 0.5).astype("int32")

# #for Categorical model
# # result = np.argmax(model.predict(input_img), axis=-1)

result = (model.predict(input_img) > 0.5).astype("int32")
print(result)