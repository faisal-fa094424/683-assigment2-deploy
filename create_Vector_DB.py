import os
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import shutil
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

VECTOR_DB_NAME = "chroma_db"
PAPERS_PATH = 'research_papers'
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OpenAI_API_KEY")

# remove any exsisiting DB to start fresh
if os.path.exists(VECTOR_DB_NAME):
    shutil.rmtree(VECTOR_DB_NAME)
os.makedirs(VECTOR_DB_NAME, exist_ok=True)


# This function to remove the refrences from the pages so they dont show in the search
def is_reference_page(text):
    """Detect bibliography / reference-list pages so they can be excluded before chunking."""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if len(lines) < 3:
        return False
    ref_pattern = re.compile(
        r'^\s*'
        r'(?:\[?\d{1,3}\]?\.?\s+[A-Z]'
        r'|https?://\S+'
        r'|doi[\s:.]+10\.\S+)',
        re.IGNORECASE,
    )
    ref_lines = sum(1 for l in lines if ref_pattern.match(l))
    return ref_lines / len(lines) > 0.4

# process all the files in specific folder chunkin, embedding amd storing in chroma DB
def process_research_library(folder_path):
    documents = []
    # using sorted so the order of files will stay consistent on every run
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(folder_path, file))
            # loader.load() automatically captures metadata: source and page
            documents.extend(loader.load())

    # Remove reference / bibliography pages before chunking
    before = len(documents)
    documents = [doc for doc in documents if not is_reference_page(doc.page_content)]
    print(f"Filtered out {before - len(documents)} reference pages, {len(documents)} content pages remaining")

    # Recursive splitting by tokens for better accuracy with technical text
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o-mini",
        chunk_size=512,
        chunk_overlap=75
    )
    chunks = text_splitter.split_documents(documents)
    return chunks



def build_vector_store(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_NAME,
        collection_name="research_papers",
    )
    return vector_db

def main():
    files_chunks = process_research_library(PAPERS_PATH)
    build_vector_store(files_chunks)

main()