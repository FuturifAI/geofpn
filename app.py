import numpy as np
import streamlit as st
from PIL import Image
import torch
import onnxruntime
st.set_option("deprecation.showfileUploaderEncoding", False)
@st.cache
def scale(x, input_space="RGB", input_range=[0,1]):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std 
    return x         
@st.cache
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
@st.cache
def cached_model():
    ort_session = onnxruntime.InferenceSession("model/geofpn.onnx")  
    return ort_session
model = cached_model()
st.title("Detect Resident Space Objects")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    DEVICE = 'cpu'
    image = np.array(Image.open(uploaded_file).convert(mode='RGB'))
    st.image(image, caption="Before", use_column_width=True, clamp=True)
    st.write("")
    st.write("Detecting Resident Space Objects...")
    with torch.no_grad():
        image_processed = scale(image).transpose(2,0,1).astype('float32')
        x_tensor = torch.from_numpy(image_processed).to(DEVICE).unsqueeze(0)
        ort_inputs = {model.get_inputs()[0].name: to_numpy(x_tensor)}
        ort_outs = model.run(None, ort_inputs)
        pr_mask_3d = np.zeros(image.shape)
        pr_mask_3d[:,:,0] = ort_outs[0][0,0,:,:]
        pr_mask_3d[:,:,1] = ort_outs[0][0,0,:,:]
        pr_mask_3d[:,:,2] = ort_outs[0][0,0,:,:]        
    st.image(pr_mask_3d, caption="After", use_column_width=True, clamp=True)
