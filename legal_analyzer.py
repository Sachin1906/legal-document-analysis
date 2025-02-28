import os
import fitz  # PyMuPDF for PDF extraction
import streamlit as st
import asyncio
import re
import spacy
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
# -----------------------------------------------------------------------------
# 🎯 Load spaCy Model
# -----------------------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------------------------------------------------------------
# 🎨 Streamlit UI Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Legal Document Analyzer", layout="wide")
st.title("📜Legal Document Analyzer")

st.markdown(
    """
    <style>
        /* Make buttons full width */
        .stButton > button {
            width: 100%;
        }
        /* Style the expander container */
        .stExpander {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 10px;
        }
        /* Optional: Increase markdown text size (you may need to adjust the selector if it doesn't work) */
        .css-1emrehy, .stMarkdown {
            font-size: 16px;
        }
        /* Set the overall page background */
        body {
            background-color: #f4f6f9;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.sidebar.image("logo.png", width=500)
st.sidebar.header("⚖️ Upload Document")

# -----------------------------------------------------------------------------
# 🔑 API Keys
# -----------------------------------------------------------------------------
groq_api_key = "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"
langsmith_api_key = "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"


if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
if langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = langsmith_api_key

# -----------------------------------------------------------------------------
# 📂 File Upload Section
# -----------------------------------------------------------------------------
uploaded_file = st.sidebar.file_uploader("📂 Upload a legal document (PDF or TXT)", type=["pdf", "txt"])

# -----------------------------------------------------------------------------
# 📜 Text Extraction Functions
# -----------------------------------------------------------------------------
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        return "\n".join(page.get_text("text") for page in doc)
    except Exception as e:
        st.error("Error extracting text from PDF: " + str(e))
        return ""

def extract_text_from_txt(txt_file):
    temp_filename = "temp_uploaded_file.txt"
    with open(temp_filename, "wb") as f:
        f.write(txt_file.read())  # Save uploaded file temporarily
    
    loader = TextLoader(temp_filename)
    documents = loader.load()
    os.remove(temp_filename)  # Delete the temporary file after loading

    return "\n".join(doc.page_content for doc in documents)

# -----------------------------------------------------------------------------
# 🔍 Legal Analysis Functions
# -----------------------------------------------------------------------------
def extract_entities(text):
    """Extracts names, dates, organizations, and locations with context."""
    doc = nlp(text)
    entities = {"Persons": [], "Dates": [], "Organizations": [], "Locations": []}
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["Persons"].append(ent.text)
        elif ent.label_ == "DATE":
            entities["Dates"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["Organizations"].append(ent.text)
        elif ent.label_ == "GPE":
            entities["Locations"].append(ent.text)
    return {key: sorted(set(value)) for key, value in entities.items()}

def extract_clause_headings(text):
    """Identifies potential contract clauses using simple heuristics."""
    lines = text.splitlines()
    clause_headings = []
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        # Heuristic: short lines that are all uppercase or end with a colon
        if len(line_stripped) < 100 and (line_stripped.isupper() or line_stripped.endswith(":") or re.match(r"^\d+\.?\s+", line_stripped)):
            clause_headings.append(line_stripped)
    return clause_headings

def classify_document_type(text):
    """Identifies the contract type based on keywords."""
    text_lower = text.lower()
    if "employment" in text_lower or "employee" in text_lower:
        return "Employment Contract"
    elif "lease" in text_lower or "landlord" in text_lower or "tenant" in text_lower:
        return "Lease Agreement"
    elif "service" in text_lower or "vendor" in text_lower:
        return "Service Agreement"
    elif "sale" in text_lower or "buyer" in text_lower or "seller" in text_lower:
        return "Sales Contract"
    else:
        return "General Contract"

def check_missing_clauses(document_type, clause_headings):
    """Checks for missing important clauses based on document type."""
    expected_clauses = {
        "Employment Contract": ["Job Description", "Compensation", "Benefits", "Termination"],
        "Lease Agreement": ["Premises", "Lease Term", "Rent", "Security Deposit", "Maintenance"],
        "Service Agreement": ["Services", "Fees", "Confidentiality", "Termination", "Liability"],
        "Sales Contract": ["Product Description", "Payment Terms", "Delivery", "Warranty"],
        "General Contract": ["Introduction", "Obligations", "Confidentiality", "Termination"]
    }
    expected = expected_clauses.get(document_type, expected_clauses["General Contract"])
    missing = [clause for clause in expected if not any(clause.lower() in heading.lower() for heading in clause_headings)]
    return missing

# -----------------------------------------------------------------------------
# 🔄 Token Counting and Summarization Setup
# -----------------------------------------------------------------------------
def count_tokens(text):
    return len(text.split())

map_prompt = ChatPromptTemplate.from_messages(
    [("system", "Write a concise summary of the following:\\n\\n{context}")]
)

async def generate_summary(content, llm):
    max_tokens = 10000
    if count_tokens(content) > max_tokens:
        return "⚠️ Chunk too large, skipping this part."
    prompt = map_prompt.invoke(content)
    response = await llm.ainvoke(prompt)
    return response.content

# -----------------------------------------------------------------------------
# 🚀 Main Processing Logic
# -----------------------------------------------------------------------------
if uploaded_file:
    st.info("📄 Processing file... Please wait.")
    
    # Extract text based on file type
    if uploaded_file.name.lower().endswith(".pdf"):
        text_content = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.lower().endswith(".txt"):
        text_content = extract_text_from_txt(uploaded_file)
    else:
        st.error("❌ Unsupported file format. Please upload a PDF or TXT file.")
        st.stop()
    
    if not text_content.strip():
        st.error("⚠️ The uploaded file is empty or unreadable.")
        st.stop()
    
    # Display a preview of the extracted text
    st.subheader("Extracted Text Preview")
    st.text_area("", text_content[:1000] + "\n...", height=200)
    
    # -----------------------------
    # Legal Analysis
    # -----------------------------
    entities = extract_entities(text_content)
    clause_headings = extract_clause_headings(text_content)
    document_type = classify_document_type(text_content)
    missing_clauses = check_missing_clauses(document_type, clause_headings)
    
    # Display Document Type
    st.subheader(f"📑 Document Type: {document_type}")
    
    # Named Entities Section with context explanation
    with st.expander("🔍 Named Entities (People, Dates, Organizations, Locations)", expanded=True):
        st.markdown("""
        **Context:** These entities help identify the key players, important dates, and locations referenced in the document.
        """)
        st.json(entities)
    
    # Clause Headings Section with explanations
    with st.expander("📌 Clause Headings & Their Significance", expanded=True):
        st.markdown("""
        **Context:** Clause headings indicate the structure of the contract. They help you quickly locate important sections.
        """)
        if clause_headings:
            for clause in clause_headings:
                st.markdown(f"- **{clause}**")
        else:
            st.write("No clear clause headings detected.")
    
    # Missing Clauses Section
    st.subheader("⚠️ Missing Clauses (Based on Expected Sections)")
    if missing_clauses:
        for clause in missing_clauses:
            st.markdown(f"❌ *{clause}* - This clause is important for a {document_type}. Consider adding it.")
    else:
        st.success("✅ All expected clauses appear to be present!")
    
    # -----------------------------
    # Summarization Process
    # -----------------------------
    st.info("⏳ Summarizing document...")
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=400, chunk_overlap=50)
    split_docs = text_splitter.split_text(text_content)
    st.write(f"🔹 Document split into {len(split_docs)} chunks for summarization.")
    
    try:
        llm = init_chat_model("llama3-8b-8192", model_provider="groq")
    except Exception as e:
        st.error("Failed to initialize LLM model: " + str(e))
        st.stop()
    
    async def summarize_documents():
        summaries = []
        for doc in split_docs:
            try:
                summary = await generate_summary(doc, llm)
                summaries.append(summary)
            except Exception as inner_e:
                summaries.append("⚠️ Error summarizing this chunk: " + str(inner_e))
            await asyncio.sleep(2)  # Delay to avoid rate limiting
        return "\n".join(summaries)
    
    try:
        final_summary = asyncio.run(summarize_documents())
    except Exception as e:
        st.error("Error during summarization: " + str(e))
        final_summary = ""
    
    st.subheader("📜 Contract Summary")
    st.write(final_summary)
