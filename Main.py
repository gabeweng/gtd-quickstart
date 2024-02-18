import os
import textwrap

import chromadb
import langchain
import openai
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, UnstructuredPDFLoader, YoutubeLoader
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from dotenv import load_dotenv
load_dotenv()

def print_response(response: str):
    print("\n".join(textwrap.wrap(response, width=100)))

model = OpenAI(temperature=0)

print(
    model(
        "You're Dwight K. Schrute from the Office. Suggest 5 places to visit in Scranton that are connected to the TV show."
    )
)