from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from apikey import apikey
import os
os.environ['OPENAI_API_KEY'] = 'youropenaiapikeyhere'
st.set_page_config(page_title="Email Response Generator")
st.header(" Email Response Generator💬")
    
    # upload file
pdf = st.file_uploader("Upload your PDF", type="pdf")
text = ""   
    # extract the text
if pdf is not None:
   pdf_reader = PdfReader(pdf)

for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
chunks = text_splitter.split_text(text)
search_index = Chroma.from_documents(chunks, OpenAIEmbeddings())
email = st.text_input("Ask for a response :")      
prompt_template = """
Respond to the email below in a way that is clear, concise, and professional. Use the style of writing that is outlined in the pdf.

Context: {context}
Topic: {email}

"""

PROMPT = PromptTemplate(
              template=prompt_template, input_variables=["context", "email"]
)

llm = OpenAI(temperature=0)

chain = LLMChain(llm=llm, prompt=PROMPT)


docs = search_index.similarity_search(email, k=4)
inputs = [{"context": doc.page_content, "topic": email} for doc in docs]
response = chain.apply(inputs)
st.write(response)
