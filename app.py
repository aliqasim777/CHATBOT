import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib
import os
import gdown

st.set_page_config(page_title="Kaggle Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 AI Chatbot (Trained on Kaggle)")
st.write("Yeh chatbot aap ki `.h5` model file par chal raha hai!")

@st.cache_resource
def download_and_load_assets():
    model_path = 'chatbot_model.h5'
    tokenizer_path = 'tokenizer.pkl'
    label_tokenizer_path = 'label_tokenizer.pkl'
    
    # 1. Model Download
    if not os.path.exists(model_path):
        model_id = '1JorqkPRoeoNGR1furN6u-SlTptqeZF8C'
        url = f'https://drive.google.com/uc?id={model_id}'
        gdown.download(url, model_path, quiet=False)
        
    # 2. Tokenizer Download
    if not os.path.exists(tokenizer_path):
        tokenizer_id = '1xYm-JLepgewm9p2j79I6jkodP5CIYGv6'
        url = f'https://drive.google.com/uc?id={tokenizer_id}'
        gdown.download(url, tokenizer_path, quiet=False)
        
    # 3. Label Tokenizer Download
    if not os.path.exists(label_tokenizer_path):
        label_id = '1Gti2c7GMVDVozp_s9xoQ-L1lURkMOg-6'
        url = f'https://drive.google.com/uc?id={label_id}'
        gdown.download(url, label_tokenizer_path, quiet=False)

    # Load Model and Tokenizers
    model = tf.keras.models.load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)
    label_tokenizer = joblib.load(label_tokenizer_path)
    
    return model, tokenizer, label_tokenizer

try:
    with st.spinner("Model Google Drive se download aur load ho raha hai..."):
        model, tokenizer, label_tokenizer = download_and_load_assets()
        reverse_word_map = {v: k for k, v in label_tokenizer.word_index.items()}
    st.success("Model kamyabi se load ho gaya hai!")
except Exception as e:
    st.error(f"Files load karne mein error aa gaya: {e}")
    st.stop()

max_length = 20
padding_type = 'post'
trunc_type = 'post'

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Yahan apna sawal likhein..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Soch raha hai..."):
            try:
                seq = tokenizer.texts_to_sequences([user_input])
                padded_seq = pad_sequences(seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
                
                prediction = model.predict(padded_seq, verbose=0)
                predicted_index = np.argmax(prediction)
                
                bot_response = reverse_word_map.get(predicted_index, "Mujhe iska jawab samajh nahi aaya.")
            except Exception as e:
                bot_response = "Maaf kijiye, prediction ke dauran koi masla ho gaya hai."

            st.markdown(bot_response)
            
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    import os
import tensorflow as tf

MODEL_PATH = 'your_model.h5'

if not os.path.exists(MODEL_PATH):
    # Use gdown to download your file from Google Drive
    import gdown
    url = 'YOUR_GOOGLE_DRIVE_PUBLIC_FILE_URL'
    gdown.download(url, MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH)
