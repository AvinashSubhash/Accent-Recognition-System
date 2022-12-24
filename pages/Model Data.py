import streamlit as st 

st.title("Project Model Details")
option = st.radio('',('CNN+LSTM','YamNet','MFCC'))
if option=="YamNet":
    st.header("Model Accuracy: 60%")
    st.header("Model Accuracy Curve: -")
    st.image("pages/model_two_accuracy.png")
    st.header("Model Loss Curve: -")
    st.image("pages/model_two_loss.png")
elif option=="CNN+LSTM":
    st.header("Model Accuracy: 50%")
    st.header("Model Accuracy Curve: -")
    st.image("pages/model_one_accuracy.png")
    st.header("Model Loss Curve: -")
    st.image("pages/model_one_loss.png")
elif option=="MFCC":
    st.header("MFCC Accuracy: 37.5%")