from langchain.llms import GPT4All, LlamaCpp # loads the type of LLM
from langchain import PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter # split characters into chunks for LLM
# from langchain.document_loaders import UnstructuredWordDocumentLoader # loads docxs
from langchain.document_loaders import UnstructuredFileLoader # loads all types of files
from langchain.document_loaders import TextLoader # load text
from langchain.document_loaders import UnstructuredURLLoader # loads URLs
from langchain.embeddings import GPT4AllEmbeddings # create embedding with GPT4ALL
# from langchain.embeddings import HuggingFaceEmbeddings # embedding
# from langchain.vectorstores import Chroma # vector storage for embeddings
from langchain.vectorstores import FAISS # vector storage for embeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA # used for question/answer by LLM
from langchain import HuggingFaceHub
from dotenv import load_dotenv
import os
load_dotenv()

def load_all_documents(path):
    """
    Load all files in the docs directory
    """
    results = []
    for filename in os.listdir(path):
        print('loaded file: ', os.path.join(path, filename))
        results.extend(load_document(os.path.join(path, filename)))
    
    return results

def load_document(file):
    """
    loads a file and split into chunks 
    """
    loader = UnstructuredFileLoader(file, strategy="fast", mode="elements")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) # token size is 1000,
    docs = text_splitter.split_documents(documents)
    return docs

def ingest_local_files(db_path, docs_path, emedding_model):
    """
    Ingest local documents, embed, push to vector store, and save on disk
    """
    db = None
    if not os.path.isdir(db_path):
        docs = load_all_documents(docs_path)
        db = FAISS.from_documents(docs, emedding_model) # embed our docs with embedding model at once and store in db
        db.save_local(db_path)
    else:
        db = load_existing_db(db_path, emedding_model)
        docs = load_all_documents(docs_path)
        new_db = FAISS.from_documents(docs, emedding_model) 
        db.merge_from(new_db)
    return db

def load_existing_db(db_path, embeddings):
    """
    load faiss db with db_path and the initialized embedding model
    """
    db = FAISS.load_local(db_path, embeddings=embeddings)
    return db

def load_LLM(model_type, model_n_ctx, llm_path):
    """
    Load the LLM
    """
    callbacks = [StreamingStdOutCallbackHandler()]
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=llm_path,
                           input={"temperature": 0.75, "max_length": 2000, "top_p": 1},
                           n_ctx=model_n_ctx, 
                           callbacks=callbacks, 
                           verbose=False)
        case "GPT4All":
            llm = GPT4All(model=llm_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        case _default:
            print(f"Model {model_type} not supported!")
            exit;
    return llm

def load_from_url(embedding_model, urls):
    """
    Loads texts from url and merges with existing db or creates db
    """
    url_loader = UnstructuredURLLoader(urls=urls)
    url_documents = url_loader.load()
    # Split into chunks
    url_text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    url_chunks = url_text_splitter.split_documents(url_documents)
    # Embed the chunks
    url_db = FAISS.from_documents(url_chunks, embedding_model) 
    return url_db

if __name__ == "__main__":
    db_path = "./db/faiss_index"
    docs_path = "./docs/"
    model_type = os.environ.get("MODEL_TYPE")
    llm_path = os.environ.get("LLM_PATH")
    model_n_ctx =  os.environ.get("MODEL_N_CTX")
    CHUNK_SIZE = os.environ.get("CHUNK_SIZE")
    CHUNK_OVERLAP = os.environ.get("CHUNK_OVERLAP")

    # Initialize embedding model
    gpt4all_embd = GPT4AllEmbeddings() 

    # URL to ingest
    urls = []

    # output FAISS index files
    local_file_index = ingest_local_files(db_path, docs_path, gpt4all_embd)
    if urls:
        url_index = load_from_url(gpt4all_embd, urls)
        # merge both FAISS index files into one
        local_file_index.merge_from(url_index)

    retriever = local_file_index.as_retriever(search_kwargs={'k': 2, 'nprobe': 10})
    llm = load_LLM(model_type, model_n_ctx, llm_path)
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", 
                                     retriever=retriever, 
                                     return_source_documents=True, verbose=True)
    
    while True:
        query = input("\nEnter a query: ")
        prompt = "Can you provide a detailed and comprehensive answer to the following question: "
        modified_query = prompt + query
        response = chain(modified_query, return_only_outputs=True)
        if query == "exit":
            break
        answer, docs = response['result'], response['source_documents']

        print(answer)
        print()
        print()
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

