from keras.models import load_model
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_hub as hub
import streamlit as st 
import numpy as np
#from tensorflow.keras.models import load_model
@tf.function
def load_16k_audio_wav(filename):
    file_content = tf.io.read_file(filename)
    audio_wav, sample_rate = tf.audio.decode_wav(file_content, desired_channels=1)
    audio_wav = tf.squeeze(audio_wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    audio_wav = tfio.audio.resample(audio_wav, rate_in=sample_rate, rate_out=16000)
    return audio_wav

@tf.function
def load_24k_audio_wav(filename):
    file_content = tf.io.read_file(filename)
    audio_wav, sample_rate = tf.audio.decode_wav(file_content, desired_channels=1)
    audio_wav = tf.squeeze(audio_wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    audio_wav = tfio.audio.resample(audio_wav, rate_in=sample_rate, rate_out=24000)
    return audio_wav

def filepath_to_embeddings(filename):
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    audio_wav = load_16k_audio_wav(filename)
    scores, embeddings, _ = yamnet_model(audio_wav)
    embeddings_num = tf.shape(embeddings)[0]
    return embeddings
def Predict(data):
    class_ = {
    0: "Irish English",
    1: "Midlands English",
    2: "Northern English",
    3: "Scottish English",
    4: "Southern English",
    5: "Welsh English",
    6: "None",
}
    reconstructed_model = load_model("second_model/model_two")
    embeddings = filepath_to_embeddings(data.name)
    #st.write(reconstructed_model.predict(embeddings))
    result = np.argmax(reconstructed_model.predict(embeddings),axis=1)
    st.header("Result: "+str(class_[np.bincount(result).argmax()]))

    