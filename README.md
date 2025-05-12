## Development of a PDF-Based Question-Answering Chatbot Using LangChain
### Reg No:212223040222
### Name:Swetha D
### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.


### PROBLEM STATEMENT:
To create a chatbot that can understand and answer questions from the content of a PDF using LangChain, making it easier to extract useful information from documents.

### DESIGN STEPS:
### STEP 1:
Load the PDF and extract the text from its pages.

### STEP 2:
Split the text into smaller chunks to make it easier to process.

### STEP 3:
Convert the text into embeddings and store them for similarity-based searching.

### STEP 4:
Ask questions, search for relevant text chunks, and return matching answers.

### PROGRAM:
```
import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("genAI.pdf")
pages = loader.load()
len(pages)
page=pages[0]
print(page.page_content[0:500])
page.metadata

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=2000,
    chunk_overlap=150,
    length_function=len
)
docs = text_splitter.split_documents(pages)
len(docs)

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()
persist_directory = 'docs/chroma/'
!rm -rf ./docs/chroma
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory=persist_directory
)
print(vectordb._collection.count())
question = "what is generative AI?"
docs = vectordb.similarity_search(question,k=3)
docs[0].page_content

```
### OUTPUT:

Document Retrieved for the Raised Question:

![image](https://github.com/user-attachments/assets/d903409f-ae99-4833-9f13-c5701848c2b4)

### RESULT:
A PDF-based chatbot was successfully created using LangChain. It was able to retrieve relevant information from the document and answer user questions accurately.
