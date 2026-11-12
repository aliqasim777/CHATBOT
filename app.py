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

# Google Drive se files download karne ka function
@st.cache_resource
def download_and_load_assets():
    # Yahan apni Google Drive ki files ke direct download links dalein
    # (Misal ke tor par gdown link format: 'https://drive.google.com/uc?id=YOUR_FILE_ID')
    
    model_path = 'chatbot_model.h5'
    tokenizer_path = 'tokenizer.pkl'
    label_tokenizer_path = 'label_tokenizer.pkl'
    
    # Agar files local folder mein mojood nahi hain toh Drive se download karein
    if not os.path.exists(model_path):
        # APNI GOOGLE DRIVE FILE ID YAHAN REPLACE KAREIN
        model_url = 'YAHAN_APNI_MODEL_FILE_KA_DRIVE_LINK_DALEN'
        gdown.download(model_url, model_path, quiet=False)
        
    if not os.path.exists(tokenizer_path):
        # APNI TOKENIZER FILE ID YAHAN REPLACE KAREIN
        tokenizer_url = 'YAHAN_APNI_TOKENIZER_FILE_KA_DRIVE_LINK_DALEN'
        gdown.download(tokenizer_url, tokenizer_path, quiet=False)
        
    if not os.path.exists(label_tokenizer_path):
        # APNI LABEL TOKENIZER FILE ID YAHAN REPLACE KAREIN
        label_url = 'YAHAN_APNI_LABEL_TOKENIZER_FILE_KA_DRIVE_LINK_DALEN'
        gdown.download(label_url, label_tokenizer_path, quiet=False)

    # Model aur Tokenizers load karna
    model = tf.keras.models.load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)
    label_tokenizer = joblib.load(label_tokenizer_path)
    
    return model, tokenizer, label_tokenizer

# Assets load karna
try:
    with st.spinner("Model aur files load ho rahi hain, bara-e-karam intezar karein..."):
        model, tokenizer, label_tokenizer = download_and_load_assets()
        reverse_word_map = {v: k for k, v in label_tokenizer.word_index.items()}
    st.success("Model kamyabi se load ho gaya hai!")
except Exception as e:
    st.error(f"Files load karne mein error aa gaya: {e}")
    st.stop()

max_length = 20
padding_type = 'post'
trunc_type = 'post'

# Chat History
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
