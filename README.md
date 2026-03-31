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
- Nginx reverse proxy
- Linode VM hosting

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

By default, the Flask app runs on:

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

This project is deployed at **qu-assigment.org** using:

- a VM hosted on **Linode**
- **Nginx** as a reverse proxy
- the Flask application running locally on port **5000**

### Production deployment flow

1. Deploy the project files to a Linux VM.
2. Create a Python virtual environment and install the dependencies.
3. Add the `.env` file with the `OPENAI_API_KEY` on the server.
4. Run `create_Vector_DB.py` to build the initial Chroma database.
5. Start the Flask app on local port `5000`.
6. Configure **Nginx** to listen on port `443` and forward incoming HTTPS requests to `127.0.0.1:5000`.


