from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers
from src.helper import *

## Load the data 

loader=DirectoryLoader('data/',glob="**/*.pdf",loader_cls=PyPDFLoader)
documents = loader.load()


## Split Text into chunks 
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                             chunk_overlap=50)
text_chunks=text_splitter.split_documents(documents=documents)


## Download the embedding models 
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                 model_kwargs={'device':'cpu'})

## Convert the Text Chunks into Embeddings and Create a FAISS Vector Store 
vector_store=FAISS.from_documents(text_chunks,embedding=embeddings)


# Initialize my llm 
llm = CTransformers(
    model=r'D:\Data Science\SimpleChatBot\model\llama-2-7b-chat.ggmlv3.q4_0.bin',
    model_type='llama',  
    config={
        'max_new_tokens': 128,
        'temperature': 0.01
    }
)

## Question Answering 
qa_prompt=PromptTemplate(template=template,input_variables=['context','question'])

## Chain 
chain=RetrievalQA.from_chain_type(llm=llm,
                                  chain_type='stuff',
                                  retriever=vector_store.as_retriever(search_kwargs={'k':2}),
                                  return_source_documents=True,
                                  chain_type_kwargs={"prompt":qa_prompt})


user_input="Tell me about words and Phrases from the paper"
result=chain({'query':user_input})
print(f"Answer:{result['result']}")