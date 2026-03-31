# Research Papers RAG App

This project is a simple research assistant for PDF papers.
It builds a Chroma vector database from all PDF files inside the `research_papers` folder, then provides a small web UI where a user can:

- ask questions about the indexed papers using RAG
- upload a new PDF paper and add it to the database
- get answers generated with OpenAI using the retrieved paper chunks

## How it works

The project has two main parts:

### 1. Initial database creation
The script `create_Vector_DB.py` scans the `research_papers` folder, loads all PDF files, removes pages that look like reference pages, splits the remaining text into chunks, and stores the chunks in a Chroma database inside `chroma_db`.

### 2. Web application
The file `app.py` runs a Flask web app.
From the UI, the user can:

- upload a new PDF file
- embed that file into the existing Chroma database
- send a query to the database
- receive a response generated with OpenAI based on the retrieved chunks

The app also supports different explanation levels in the prompt, so answers can be adjusted for different audiences.

## Main technologies

- Flask
- LangChain
- ChromaDB
- OpenAI Embeddings (`text-embedding-3-small`)
- OpenAI Chat model (`gpt-4o-mini`)
- PyMuPDF for PDF loading

## Project structure

```text
.
├── app.py
├── create_Vector_DB.py
├── requirements.txt
├── research_papers/
└── chroma_db/
```

## Setup

### 1. Clone the project

```bash
git clone <your-repo-url>
cd <your-project-folder>
```

### 2. Create and activate a virtual environment

On Linux or macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add environment variables
Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the app

### Step 1: Build the vector database

```bash
python create_Vector_DB.py
```

This will read all PDFs from `research_papers/` and create a fresh Chroma database in `chroma_db/`.

### Step 2: Start the Flask app

```bash
python app.py
```

By default, the app runs on:

```text
http://localhost:5000
```

## Using the app

1. Place your research papers as PDF files inside `research_papers/`.
2. Run `create_Vector_DB.py` to create the initial vector database.
3. Start the Flask app with `app.py`.
4. Open the UI in your browser.
5. Ask a question about the papers, or upload a new PDF.

## Deployment

This project can be deployed on any service that supports Python web apps.

### Option 1: Simple server or VM

Install the dependencies, set the `OPENAI_API_KEY`, build the Chroma database, and run:

```bash
python app.py
```

For production, it is better to run Flask behind a production server such as Gunicorn.

Example:

```bash
gunicorn -b 0.0.0.0:5000 app:app
```

### Option 2: Streamlit Cloud or similar platforms

This current project is written with Flask, so it is better suited to platforms that support general Python web apps.
If deployment on Streamlit Cloud is required, the UI would need to be rewritten in Streamlit first.

## Notes

- The script removes pages that appear to be bibliography/reference pages before embedding.
- Uploaded PDFs are also added to the `research_papers` folder after successful processing.
- The current retrieval uses similarity search from Chroma.
- The current app expects the HTML template `index.html` to exist in a `templates/` folder.

## Future improvements

- add hybrid retrieval (dense + BM25)
- add configurable top-k retrieval from the UI
- add source citations directly in the displayed answer
- add duplicate-file detection by content hash
- add Docker support for easier deployment

