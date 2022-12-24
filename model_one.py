import keras
import streamlit as st
import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from pathlib import Path
import PIL
from PIL import Image
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

def ConvertAudio(data,filename):
  input_data = read(data)
  filename = filename.split(".")[0]
  audio = input_data[1]
  # plot the first 1024 samples
  #plt.figure(figsize=(10, 5))
  plt.plot(audio)
  # label the axes
  #plt.ylabel("Amplitude")
  #plt.xlabel("Time")
  # set the title  
  #plt.title("Sample Wav")
  # display the plot
  fig1 = plt.gcf()
  plt.axis('off')
  plt.gray()
  #plt.show()
  #plt.draw()
  st.pyplot(fig1)
  fig1.savefig("generated-images/"+str(filename)+"-image.jpg")
  plt.close()

def Predict(data,text):
    class_ ={0:'welsh_english', 1:'northern_english', 2:'southern_english',3:'irish_english', 4:'scottish_english', 5:'midlands_english'}
    reconstructed_model = keras.models.load_model("image-text-model")
    #_ = Path("input-audio/",data.name)
    ConvertAudio(data,data.name)
    image = Image.open("generated-images/"+data.name.split(".")[0]+"-image.jpg")
    image = image.resize((150,150))
    image_data = np.asarray(image)
    st.write("Image data array shape: ",image_data.shape)
    text_data = text.split(" ")
    text_data = [a.lower() for a in text_data[1:]]
    texts=[]
    texts_all=[]
    texts.append(' '.join(text_data))
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(texts_all)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    #print('Found {} unique tokens'.format(len(word_index)))
    tag_data = pad_sequences(sequences, maxlen=100)
    st.write("Final text_input data shape: ",tag_data.shape)
    a = np.expand_dims(np.array(image_data),0)
    res = reconstructed_model.predict([a,np.array(tag_data)])
    st.write("Model Prediction: \n")
    st.header("Result: "+str(class_[np.argmax(res)]))
    #for i in range(len(res[0])):
    #   st.header("Class "+str(i)+": "+str(res[0][i]))
    #st.write("Data recieved")

    