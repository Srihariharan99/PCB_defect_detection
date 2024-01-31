

import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from torch.nn.functional import relu
import cv2
import numpy as np

st.set_page_config(page_title="Autoencoder", page_icon="",layout="wide",initial_sidebar_state="expanded")
st.title("Autoencode for PCB Board")

model_path = "autoencoder_tensor_to_torch_v0.2.pt"
SIZE = [640,256]



with st.sidebar:
    # Adding header to sidebar
    st.header("Please upload a image")     
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
# Creating two columns on the main page
col1, col2 ,col3= st.columns(3)

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = Image.open(source_img)
        st.image(source_img,caption="Uploaded Image",use_column_width=True)
        #st.write(source_img.shape)
try:
    model = torch.jit.load(model_path, map_location=torch.device('cpu'))
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

clicked = st.sidebar.button('Detect Objects')

def defect(def_image):
  file_bytes = np.asarray(bytearray(def_image.read()), dtype=np.uint8)
  image_bgr= cv2.imdecode(file_bytes, 1)


  image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
  image1 = cv2.resize(image, (SIZE[1], SIZE[0]))
  image = image1.astype('float32') / 255.0
  image = torch.tensor(image)
  batch = image.unsqueeze(0)
  image = batch.permute(0,3,1,2)

  with torch.no_grad():
    out = model(image)




  output= out
  output=output.permute(0,2,3,1)
  out = output.numpy()*255
  outputs = out[0].astype("uint8")
  inputs = image1
  outputs = cv2.resize(outputs,(256, 640))


  sub= cv2.absdiff(inputs,outputs)
  gray = cv2.cvtColor(sub,cv2.COLOR_RGB2GRAY)
  _,thersh = cv2.threshold(gray,20,120,cv2.THRESH_BINARY)
  kernel = np.ones((2,2), np.uint8)
  erosion= cv2.erode(thersh, kernel)  

  kernel = np.ones((5, 5), np.uint8)
  dilation = cv2.dilate(erosion,kernel)
  edge = cv2.Canny(dilation,65,120)
  contours, hierarchy = cv2.findContours(edge,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  for con in contours:
          x,y,w,h = cv2.boundingRect(con)
          cv2.rectangle(inputs,(x,y),(x+w,y+h),(255,0,0),2)
          cv2.rectangle(outputs,(x,y),(x+w,y+h),(255,0,0),1)
  return inputs,outputs

if clicked:
     
  result = defect(source_img)

  with col2:
          st.image(result[0],
                  caption='Detected find Image',
                  use_column_width=True
                  )
  with col3:
        st.image(result[1],
                caption='Regenerated image',
                use_column_width=True
                )








