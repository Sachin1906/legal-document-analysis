import os
import fitz  # PyMuPDF for PDF extraction
import streamlit as st
import asyncio
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter

# Streamlit UI Title
st.title("📄 Legal Document Summarizer")

# User Input for API Keys
groq_api_key = st.text_input("🔑 Enter Groq API Key:", type="password")
langsmith_api_key = st.text_input("🔑 Enter LangSmith API Key:", type="password")

# Set API Keys
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
if langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = langsmith_api_key

# File Uploader
uploaded_file = st.file_uploader("📂 Upload a PDF or TXT file", type=["pdf", "txt"])

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# Function to extract text from a TXT file
def extract_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")

# Function to count tokens (Simple Approximation)
def count_tokens(text):
    return len(text.split())  # Rough estimate (1 word ≈ 1.2 tokens)

# ✅ Corrected ChatPromptTemplate usage
map_prompt = ChatPromptTemplate.from_messages(
    [("system", "Write a concise summary of the following:\\n\\n{context}")]
)

# Summarization Function
async def generate_summary(content, llm):
    max_tokens = 5500  # Keep below Groq’s 6000 TPM limit
    if count_tokens(content) > max_tokens:
        return "⚠️ Chunk too large, skipping this part."

    prompt = map_prompt.invoke(content)
    response = await llm.ainvoke(prompt)
    return response.content

# Process file and summarize
if uploaded_file and groq_api_key:
    st.info("📄 Processing file... Please wait.")

    # Determine file type and extract text
    if uploaded_file.name.endswith(".pdf"):
        text_content = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".txt"):
        text_content = extract_text_from_txt(uploaded_file)
    else:
        st.error("❌ Unsupported file format. Please upload a PDF or TXT file.")
        st.stop()

    # Handle Empty File
    if not text_content.strip():
        st.error("⚠️ The uploaded file is empty. Please upload a valid document.")
        st.stop()

    # Split text into chunks (Smaller chunks to avoid exceeding Groq’s TPM limit)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=50)
    split_docs = text_splitter.split_text(text_content)
    st.write(f"🔹 Split document into {len(split_docs)} chunks.")

    # Initialize LLM Model
    llm = init_chat_model("llama3-8b-8192", model_provider="groq")

    # Async Summarization Function with Rate Limit Handling
    async def summarize_documents():
        summaries = []
        for doc in split_docs:
            summary = await generate_summary(doc, llm)
            summaries.append(summary)
            await asyncio.sleep(3)  # ⏳ 3-second delay to prevent rate limit errors
        return "\n".join(summaries)

    # Run async safely in Streamlit
    final_summary = asyncio.run(summarize_documents())

    # Display summary
    st.subheader("📌 Summary")
    st.write(final_summary)
