import streamlit as st
import os
import pdfplumber
from functools import lru_cache

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from openai import OpenAI

# ---------------------------
# OpenRouter Client
# ---------------------------

client = OpenAI(
    api_key=st.secrets["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

# ---------------------------
# Read PDF
# ---------------------------

def read_pdf(file_path):
    text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


# ---------------------------
# Create Vector Database
# ---------------------------

@lru_cache(maxsize=1)
def load_vectorstore():

    pdf_path = "college.pdf"

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            "college.pdf not found. Put it in the same folder as app.py"
        )

    text = read_pdf(pdf_path)

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    return vectorstore


# ---------------------------
# Chat Function
# ---------------------------

def chat_with_bot(query):

    if query.lower() in ["hi", "hello", "hey"]:
        return "Hello! 👋 How can I help you with college admissions today?"

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a helpful college admission assistant.

Use ONLY the provided context.

If the answer is not available in the context, reply:

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


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(
    page_title="College Admission Chatbot",
    page_icon="🎓"
)

st.title("🎓 College Admission Chatbot")

st.write(
    "Ask questions about admissions, fees, eligibility, courses, placements and more."
)

if not os.path.exists("college.pdf"):
    st.error(
        "college.pdf not found. Upload it to your GitHub repository."
    )
    st.stop()

question = st.text_input(
    "Enter your question:"
)

if st.button("Ask"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:

        with st.spinner("Thinking..."):

            try:

                answer = chat_with_bot(question)

                st.success(answer)

            except Exception as e:

                st.error(f"Error: {str(e)}")
