import time

import streamlit as st

from main_pg import chat

st.title("RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

def stream_response(prompt):
    response_placeholder = st.empty()
    full_response = chat(prompt)
    streamed_response = ""

    for word in full_response.split():
        streamed_response = f"{streamed_response} {word}".strip()
        response_placeholder.markdown(streamed_response + "▌")
        time.sleep(0.01)

    response_placeholder.markdown(full_response)
    return full_response

# Purane messages dikhao
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Naya input lo
if prompt := st.chat_input("Apna sawal poochein..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        response = stream_response(prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})