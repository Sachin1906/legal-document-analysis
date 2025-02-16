import os
import getpass
import fitz  # PyMuPDF
from langchain_community.document_loaders import TextLoader
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

# Enable LangSmith (Optional)
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter LangSmith API Key: ")

# Set Groq API Key
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

# Initialize LLM Model
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# Function to extract text from a TXT file
def extract_text_from_txt(txt_path):
    loader = TextLoader(txt_path)
    documents = loader.load()
    return "\n".join(doc.page_content for doc in documents)

# Ask user for file input
file_path = input("Enter the path to your PDF or TXT file: ")

# Determine file type and extract text
if file_path.lower().endswith(".pdf"):
    text_content = extract_text_from_pdf(file_path)
elif file_path.lower().endswith(".txt"):
    text_content = extract_text_from_txt(file_path)
else:
    raise ValueError("Unsupported file format. Please provide a PDF or TXT file.")

# Split text into chunks
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_text(text_content)
print(f"Generated {len(split_docs)} document chunks.")

# Summarization Prompt
map_prompt = ChatPromptTemplate.from_messages(
    [("system", "Write a concise summary of the following:\\n\\n{context}")]
)

# Summarization Function
async def generate_summary(content):
    prompt = map_prompt.invoke(content)
    response = await llm.ainvoke(prompt)
    return response.content

# Process each chunk and summarize
import asyncio

async def summarize_documents():
    summaries = await asyncio.gather(*[generate_summary(doc) for doc in split_docs])
    final_summary = "\n".join(summaries)
    return final_summary

# Run summarization
final_summary = asyncio.run(summarize_documents())

# Print summary
print("\nFinal Summary:")
print(final_summary)
