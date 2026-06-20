import streamlit as st
import os
from functools import lru_cache
import pdfplumber

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# OpenRouter client
client = OpenAI(
    api_key=st.secrets["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

# Read PDF
def read_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Load and cache vector database
@lru_cache(maxsize=1)
def load_vectorstore():
    pdf_path = "college.pdf"

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            "college.pdf not found. Place the PDF in the same folder as app.py"
        )

    text = read_pdf(pdf_path)

    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.create_documents([text])

    embeddings = OpenAIEmbeddings(
    api_key=st.secrets["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

# Chat function
def chat_with_bot(query):

    if query.lower() in ["hi", "hello", "hey"]:
        return "Hello! 👋 How can I help you with college admissions today?"

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    docs = retriever.invoke(query)

    context = "\n---\n".join(
        [doc.page_content for doc in docs]
    )

    if not context.strip():
        return "I don't have that information. Please contact the admission office."

    prompt = f"""
You are a college admission assistant.

Answer only from the provided context.

Keep the answer:
- Short
- Accurate
- Friendly

If information is not available, reply:
"I don't have that information. Please contact the admission office."

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content.strip()

# ---------------- STREAMLIT UI ----------------

st.set_page_config(
    page_title="College Admission Chatbot",
    page_icon="🎓"
)

st.title("🎓 College Admission Chatbot")

st.write("Ask questions about admissions, fees, eligibility, courses, etc.")

if not os.path.exists("college.pdf"):
    st.error("college.pdf not found. Put the PDF in the same folder as app.py")
    st.stop()

query = st.text_input(
    "Enter your question:"
)

if st.button("Ask"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                answer = chat_with_bot(query)
                st.success(answer)

            except Exception as e:
                st.error(f"Error: {e}")
