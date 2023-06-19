import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
def main():
    os.environ['OPENAI_API_KEY'] = ''
    st.set_page_config(page_title="Email Response Generator")
    st.header("Email Response Generator 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
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
      
      # create embeddings
      embeddings = OpenAIEmbeddings()
      search_index = FAISS.from_texts(chunks, embeddings)
      
      # show user input
      email = st.text_input("Ask for a response :")      
      prompt_template = """
Respond to the email below in a way that is clear, concise, and professional. Use the style of writing that is outlined in the pdf.

Context: {context}
Email: {email}

"""

      PROMPT = PromptTemplate(
              template=prompt_template, input_variables=["context", "email"]
)

      llm = OpenAI(temperature=0)

      chain = LLMChain(llm=llm, prompt=PROMPT)

      def generate_response(email):
          docs = search_index.similarity_search(email, k=4)
          inputs = [{"context": chunks, "email": email} for doc in docs]
          response = chain.apply(inputs)
      res = generate_response(email)
      st.write(res)
    

if __name__ == '__main__':
    main()
