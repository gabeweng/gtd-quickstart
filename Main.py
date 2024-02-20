import os
import textwrap

import chromadb
import langchain
import openai
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma

from dotenv import load_dotenv
load_dotenv()

def print_response(response: str):
    print("\n".join(textwrap.wrap(response, width=100)))

model = OpenAI(temperature=0, model="gpt-3.5-turbo-1106")

print(
    model(
        "You're Dwight K. Schrute from the Office. Suggest 5 places to visit in Scranton that are connected to the TV show."
    )
)

#Q&A Over a Document

loader = WebBaseLoader(
    "https://blog.twitter.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm"
)
documents = loader.load()
document = documents[0]
index = VectorstoreIndexCreator().from_loaders([loader])
query = """
You're Dwight K. Schrute from the Office.
Explain the Twitter recommendation algorithm in 5 sentences using analogies from the Office.
"""
print_response(index.query(query))

#Using a Prompt Template

template = """You're Dwight K. Schrute from the Office.

{context}

Answer with analogies from the Office to the question and the way Dwight speaks.

Question: {question}
Answer:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
print(
    prompt.format(
        context="Paper sells are declining 10% year over year.",
        question="How to sell paper?",
    )
)