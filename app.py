import streamlit as st
from dotenv import load_dotenv
import pickle
import PyPDF2
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from langchain.vectorstores import FAISS
#from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings


 
# Sidebar contents
with st.sidebar:
    st.title('💊 Evidence Based Medicine')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Muhammad Ammar](https://www.linkedin.com/in/muhammad-ammar12/)')
 
load_dotenv()

def custom_prompt():
  from langchain.prompts import PromptTemplate
  template_det ="""
  Based on the knowledge provided to you. You are expected to provide answer based on PICO framework.  You are a Professional medical scientist who have every knowledge in the medical field to answer user {context}. your answer should include description in detail each element about population (P), Intervention (I), Comparison (C) and Outcome (O) as stated in report/paper. What description should include is further guided. list each point separately.

  Population (P) description: "Setting (including location and social context), Inclusion criteria, Exclusion criteria, Method of recruitment of participants (e.g. phone, mail, clinic patients), consent was taken or not, Total no. randomised (or total pop. at start of study for NRCTs),
  Clusters (if applicable, no., type, no. people per cluster), Baseline imbalances, Withdrawals and exclusions (if not provided below by outcome), age, sex, race/ethnicity, Severity of illness,
  Co-morbidities, Other relevant sociodemographics, Subgroups measured, Subgroups reported, and Notes."

  Intervention (I) description: "Group name, No. randomised to group (specify whether no. people or clusters), Theoretical basis (include key references), Description (include sufficient detail for replication, e.g. content, dose, components), Duration of treatment period, Timing (e.g. frequency, duration of each episode), 
  Delivery (e.g. mechanism, medium, intensity, fidelity), Providers (e.g. no., profession, training, ethnicity etc. if relevant), Co-interventions, Economic information (i.e. intervention cost, changes in other costs as result of intervention), Resource requirements (e.g. staff numbers, cold chain, equipment), Integrity of delivery, Compliance."

  Comparison (C) description: "Group name, No. randomised to group (specify whether no. people or clusters), Theoretical basis (include key references), Description (include sufficient detail for replication, e.g. content, dose, components), Duration of treatment period, Timing (e.g. frequency, duration of each episode), 
  Delivery (e.g. mechanism, medium, intensity, fidelity), Providers (e.g. no., profession, training, ethnicity etc. if relevant), Co-interventions, Economic information (i.e. intervention cost, changes in other costs as result of intervention), 
  Resource requirements (e.g. staff numbers, cold chain, equipment), Integrity of delivery, Compliance."

  Outcome (O) description: "Outcome name, Time points measured (specify whether from start or end of intervention), Time points reported, Outcome definition (with diagnostic criteria if relevant), Person measuring/ reporting, Unit of measurement  (if relevant), Scales: upper and lower limits (indicate whether high  or low score is good), Is outcome/tool validated?,
  Imputation of missing data (e.g. assumptions made for ITT analysis), Assumed risk estimate (e.g. baseline or population risk noted  in Background), Power (e.g. power & sample size calculation, level of power achieved)."

  Once you have provided the PICO description as per report/paper in detail, must provide answer to the user (doctor, researcher or scientist) {context} which should be backed by evidance from PICO which you extracted earlier in a precise paragraph and tell what was the population, which intervention and comparisons were used and what was the outcome. ignore words like maybe or any other which may create ambiguity for users you answer should be solid based on evidance.    


  """
  Prompt=PromptTemplate(
      input_variables=["context"],
      template=template_det

  )
  return Prompt

def open_llm():
  from langchain_community.chat_models import ChatOpenAI
  chat_model = ChatOpenAI(model_name='gpt-4o')
  return chat_model

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    num_pages = len(pdf_reader.pages)
    text = ''
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text
def suggestions(text):
  llm=open_llm()
  pro="""Here is explaining of each component of the PICO framework.
Patient/Population/Problem (P):

"This component typically refers to the specific group of individuals, patient population, or the problem being addressed. Provide details on how defining this aspect helps in structuring research questions and clinical inquiries."

Intervention (I):

"This component represents the intervention, treatment, therapy, or action being considered for the patient population or problem identified in the 'P' component. Explain how this element is crucial in formulating research hypotheses and guiding evidence-based practices."

Comparison (C):

"This aspect involves comparing interventions or conditions and may include placebos, alternative treatments, or standard care. Detail how the comparison element contributes to making research questions more specific and aids in drawing meaningful conclusions from research studies."

Outcome (O):

"This component refers to the desired outcome or result that researchers aim to achieve as a consequence of the intervention. Explain how defining the expected outcomes helps in setting measurable goals for research and guides decision-making in clinical settings."

  
  You are a helpful assistant for doctors to genrate PICO based questions for {context}. write a short summary of the medical research paper, which should focus on the importance of the paper.
  Then generate three PICO based Questions in bullet points. Note if the text is not related to medical just tell user nicely to upload a medical paper.  """ 

  from langchain.prompts import PromptTemplate
  from langchain.chains import LLMChain
  Prompt=PromptTemplate(
    input_variables=["context"],
    template=pro)
  chain=LLMChain(llm=llm, prompt=Prompt)
  res=chain.run(text)
  return res

import textwrap

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    st.write(wrap_text_preserve_newlines(llm_response['result']))
    
    with st.expander("Click here to see source chunks"):
      for source in llm_response["source_documents"]:
        st.write(source)

    
            
              

def main():
    st.header("Query your document 🤔💭")
 
 
    # upload a PDF file
    uploaded_files = st.file_uploader("Upload your PDF to ask PICO based questions", type='pdf',accept_multiple_files=True)
 
    # st.write(pdf)
    if uploaded_files is not None:
        check=0
        for uploaded_file in uploaded_files:
            file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
            #st.write(file_details)
            text = read_pdf(uploaded_file)
 
            

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=0,
                length_function=len
            )

            summary_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=0,
                length_function=len
            )

            summary_text=summary_splitter.split_text(text=text)


            chunks = text_splitter.split_text(text=text)

            st.write("Total number of chunks is : ",len(chunks))
            with st.expander("Click to see text chunks"):
              st.write(chunks)
              
            res=suggestions(summary_text[0])
            st.write(res)
            
            instructor_embeddings = OpenAIEmbeddings()
            db_instructEmbedd = FAISS.from_texts(chunks, instructor_embeddings)
            retriever = db_instructEmbedd.as_retriever(search_kwargs={"k": 30})
            #retriever.search_type='mmr'
            #st.write("Done bae")
            placeholder = st.empty()

            # Widget to be positioned at the bottom
            with placeholder:
              query = st.text_input("Ask questions about your PDF file:", key=check)
            #query = st.text_input("Ask questions about your PDF file:")
            # st.write(query)
            check=check+1
            if query:
                docs = db_instructEmbedd.similarity_search(query=query, k=60)
                
                o_llm = open_llm()
                Prompt=custom_prompt()
                #chain = load_qa_chain(llm=g_llm, chain_type="stuff")
                qa_chain = RetrievalQA.from_chain_type (llm=o_llm,
                                  chain_type="stuff",
                                  chain_type_kwargs={"prompt":Prompt},
                                  retriever=retriever,
                                  return_source_documents=True)
                with get_openai_callback() as cb:
                    response = qa_chain(query)
                    print(cb)
                process_llm_response(response)
 
      
 
if __name__ == '__main__':
    main()