import streamlit as st
from app import chat_with_bot

# Page config
st.set_page_config(page_title="AI Admission Assistant", page_icon="🎓")

st.title("🎓 AI College Admission Assistant")

# Store messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
user_input = st.chat_input("Ask your question...")

# Only run if user typed something
if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Build conversation history (last 5 messages only)
    history = "\n".join(
        [f"{m['role']}: {m['content']}" for m in st.session_state.messages[:-1][-5:]]
    )

    # Get bot response
    with st.spinner("Thinking..."):
        reply = chat_with_bot(
            user_input + "\n\nPrevious conversation:\n" + history
        )

    # Show bot response
    st.chat_message("assistant").markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})