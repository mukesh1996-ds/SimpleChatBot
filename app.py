import streamlit as st
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers

# --- Optional helper import if you have a custom prompt in `src/helper.py`
try:
    from src.helper import template
except ImportError:
    template = """Use the following context to answer the question.\nContext: {context}\n\nQuestion: {question}\nAnswer:"""

# --- Page Title
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("📄 Chat with your PDF using LLaMA and LangChain")

# --- Sidebar: User input for prompt
user_input = st.text_input("Ask a question from your PDF:", "")

# --- Caching to avoid reloading everything every time
@st.cache_resource
def load_vector_store():
    loader = DirectoryLoader('data/', glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )

    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
    return vector_store

@st.cache_resource
def load_llm():
    return CTransformers(
        model=r'D:\Data Science\SimpleChatBot\model\llama-2-7b-chat.ggmlv3.q4_0.bin',
        model_type='llama',
        config={
            'max_new_tokens': 128,
            'temperature': 0.01
        }
    )

# --- Load models
with st.spinner("Loading model and vector store..."):
    vector_store = load_vector_store()
    llm = load_llm()

# --- Setup QA chain
qa_prompt = PromptTemplate(template=template, input_variables=["context", "question"])
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_prompt}
)

# --- Run chain on user input
if user_input:
    with st.spinner("Generating answer..."):
        result = qa_chain({"query": user_input})
        st.subheader("Answer")
        st.write(result['result'])

        # Optionally show source documents
        with st.expander("Source Documents"):
            for doc in result['source_documents']:
                st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                st.write(doc.page_content[:500])  # Show only first 500 characters
