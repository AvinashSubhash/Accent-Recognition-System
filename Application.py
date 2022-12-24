import streamlit as st
import MFCC.runner as rn
import model_one
import model_two
st.title("Accent Recognition & Evaluation System")
audio_file = st.file_uploader("Upload the audio file")
if audio_file:
    option = st.radio("Select the model that you want to use: ",('CNN+LSTM','YamNet','MFCC'))
    if option == 'CNN+LSTM':
        text_data = st.text_input("Enter the text spoken in the audio")
        if text_data:
            model_one.Predict(audio_file,text_data)
    elif option == 'YamNet':
        model_two.Predict(audio_file)
    elif option == 'MFCC':
        result = rn.play_with_me(audio_file.name)
        st.header("Result: "+str(result[1]))
        st.write("Scores: ",result[0])


