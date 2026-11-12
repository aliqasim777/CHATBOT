import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib
import os

# Page Configuration
st.set_page_config(page_title="Kaggle Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 AI Chatbot (Trained on Kaggle)")
st.write("Yeh chatbot aap ke train kiye gaye model aur `.h5` file par chal raha hai!")

# Model aur Tokenizers load karne ka function (Cache ke sath taake bar bar load na ho)
@st.cache_resource
def load_chatbot_assets():
    # Model load karna
    model = tf.keras.models.load_model('chatbot_model.h5')
    
    # Tokenizers load karna
    tokenizer = joblib.load('tokenizer.pkl')
    label_tokenizer = joblib.load('label_tokenizer.pkl')
    
    return model, tokenizer, label_tokenizer

# Assets load karna
try:
    model, tokenizer, label_tokenizer = load_chatbot_assets()
    reverse_word_map = {v: k for k, v in label_tokenizer.word_index.items()}
    st.success("Model aur Tokenizers kamyabi se load ho gaye hain!")
except Exception as e:
    st.error(f"Files load karne mein error aa gaya: {e}")
    st.stop()

max_length = 20
padding_type = 'post'
trunc_type = 'post'

# Session State mein chat history maintain karna
if "messages" not in st.session_state:
    st.session_state.messages = []

# Purani chat history screen par dikhana
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User ka input lena
if user_input := st.chat_input("Yahan apna sawal likhein..."):
    # User ka message add karna
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Bot ki response generate karna
    with st.chat_message("assistant"):
        with st.spinner("Soch raha hai..."):
            try:
                # Text ko sequence aur padding dena
                seq = tokenizer.texts_to_sequences([user_input])
                padded_seq = pad_sequences(seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
                
                # Model prediction
                prediction = model.predict(padded_seq, verbose=0)
                predicted_index = np.argmax(prediction)
                
                # Jawab dhoondna
                bot_response = reverse_word_map.get(predicted_index, "Mujhe iska jawab samajh nahi aaya.")
            except Exception as e:
                bot_response = "Maaf kijiye, prediction ke dauran koi masla ho gaya hai."

            st.markdown(bot_response)
            
    # Bot ka message history mein save karna
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
