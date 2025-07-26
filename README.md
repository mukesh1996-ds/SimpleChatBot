````markdown
# 📄 PDF ChatBot with RAG (Retrieval-Augmented Generation)

A powerful AI-powered chatbot system built on top of PDF document understanding using **Langchain**, **Hugging Face embeddings**, **FAISS vector store**, and **CTransformers**. The application enables intelligent querying of PDF documents via a simple **Streamlit** interface.

## 🚀 Features

- 📁 Upload and query your **PDF documents**
- 🤖 **Retrieval-Augmented Generation (RAG)** based QA pipeline
- 🧠 Embeddings from Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`)
- ⚡ Fast similarity search using **FAISS**
- 🧩 Modular and extensible with **Langchain** and `CTransformers`
- 🖥️ Intuitive and interactive **Streamlit** web app interface

---

## 🛠️ Tech Stack

- **Python**
- **Langchain**
- **Hugging Face Transformers**
- **FAISS**
- **CTransformers** (LLaMA/GGML based local model)
- **Streamlit**

---

## 📁 Project Structure

```bash
├── data/                   # Folder containing PDF documents
├── src/
│   ├── helper.py          # Utility functions
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md              # You're here!
````

---

## 🔧 Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/pdf-chatbot.git
   cd pdf-chatbot
   ```

2. **Create virtual environment & activate**

   ```bash
   python -m venv venv
   source venv/bin/activate  # on Linux/Mac
   venv\Scripts\activate     # on Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

---

## 🧠 How It Works

1. **Document Loading**:
   Uses `DirectoryLoader` and `PyPDFLoader` from Langchain to load and parse PDFs.

2. **Text Splitting**:
   Splits long documents into manageable chunks using `RecursiveCharacterTextSplitter`.

3. **Embeddings**:
   Converts text chunks into embeddings via Hugging Face models (MiniLM).

4. **Vector Storage**:
   Stores embeddings in a FAISS index for fast similarity search.

5. **LLM Integration**:
   Uses `CTransformers` with a locally hosted LLaMA model to generate answers based on retrieved chunks.

6. **QA Chain**:
   RetrievalQA chain from Langchain integrates the retriever and the LLM.

---

## 📸 Demo Screenshot

![PDF Chatbot Streamlit UI](https://via.placeholder.com/800x400.png?text=PDF+ChatBot+Demo)

---

## 📄 Example Usage

1. Upload your PDF files into the `data/` folder.
2. Launch the Streamlit app.
3. Ask questions based on your PDF content — get instant, accurate answers!

---

## 🧪 Example Code Snippet

```python
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from src.helper import *

loader = DirectoryLoader('data/', glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
...
```

