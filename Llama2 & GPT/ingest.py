from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

DATA_PATH="data/"
DB_FAISS_PATH="vectorstore/db_faiss"

def create_vector_db():
    loader=DirectoryLoader(DATA_PATH,glob="*.pdf",loader_cls=PyMuPDFLoader) 
    # load all pdfs in DATA_PATH and glob is used to filter the files , i.e it only accepts pdf files and 
    # loader_cls is used to load the pdfs
    documents=loader.load()
    
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
     # split the text into chunks of 500 characters with 50 characters overlap , 
    # the overlap means that the last 50 characters of the previous chunk will be the 
    # first 50 characters of the next chunk
    
    texts=text_splitter.split_documents(documents)

    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                     model_kwargs={'device':'cpu'})
    
    db=FAISS.from_documents(texts,embeddings)
    #here the texts contains the splitted texts and embeddings is the model used to convert the text into vectors

    db.save_local(DB_FAISS_PATH)

if __name__=="__main__":
    create_vector_db()