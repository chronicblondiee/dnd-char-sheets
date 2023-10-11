from langchain.document_loaders import OnlinePDFLoader
from langchain.document_loaders import UnstructuredPDFLoader # offline documents
from langchain.document_loaders import DirectoryLoader
#from pdfminer.pdfinterp import PDFResourceManager
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
import sys
import os

class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# load the pdf and split it into chunks
#loader = OnlinePDFLoader("http://media.wizards.com/2018/dnd/downloads/DnD_BasicRules_2018.pdf")
#loader = UnstructuredPDFLoader(file_path=["/opt/ollama/mistral/langchain/DnD-Handbooks/DnD_BasicRules_2018.pdf", "/opt/ollama/mistral/langchain/DnD-Handbooks/Players_Handbook_5e.pdf", "/opt/ollama/mistral/langchain/DnD-Handbooks/Dungeon_Masters_Guide_5e.pdf","/opt/ollama/mistral/langchain/DnD-Handbooks/Monsters_Manual_5e.pdf",  "/opt/ollama/mistral/langchain/DnD-Handbooks/Moon_over_Graymoor.pdf"])
loader = DirectoryLoader('/opt/ollama/dnd-char-sheets/source-data/', glob="**/*.pdf", use_multithreading=True, show_progress=True)
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

with SuppressStdout():
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

while True:
    query = input("\nQuery: ")
    if query == "exit":
        break
    if query.strip() == "":
        continue

    # Prompt
    template = """You are a highly experienced Dungeons & Dragons player.
    Use the following pieces of context to create Dungeons & Dragons character sheets.
    Dungeons & Dragons character sheets, allow you choose many different classes races and spells, pick the best suited attributes or suggest them based on class.
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    llm = Ollama(model="mistral:latest", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    result = qa_chain({"query": query})