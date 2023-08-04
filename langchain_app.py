import streamlit as st
import streamlit_toggle as tog
import pandas as pd

from langchain.callbacks import StreamlitCallbackHandler

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS 

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Set page title
st.set_page_config(page_title="Ask the Behaviourial Science Guru 🤖", page_icon="🤖", layout="wide")
st.title("Ask the Behavioural Science Guru 🤖")

# Import pdf file and generate embeddings
def load_pdf(input_pdf, API_KEY):
    try:
        doc_reader = PdfReader(input_pdf)
    except:
        st.error("Cannot load pdf file")
    
    # read data from the file and put them into a variable called raw_text
    raw_text = ''
    for i, page in enumerate(doc_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    
    # Splitting up the text into smaller chunks for indexing
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200, #striding over the text
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    docsearch = FAISS.from_texts(texts, embeddings)
    return docsearch

def fix_langchain_json(json):
    if json[-3:] != "```":
        json = json + "`"
    
    index = json.find('\n\t"summary":')
    if index == -1 or index == 0 or "," in json[index-5:index]:
        return json
    return json[:index] + "," + json[index:]


def generate_summary(docsearch, API_KEY):
    authors_schema = ResponseSchema(name="authors",
                             description="Who was this document authored or prepared by? \
                             List the names of authors or answer Unknown if it cannot be determined\
                             Do not include people listed only under acknowledgements")
    summary_schema = ResponseSchema(name="summary",
                             description="Give a summary of the\
                             content of this document, including\
                             key takeaways, conclusions,\
                             and the purpose of the document")
    response_schemas = [authors_schema, summary_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    summary_template = """\
    For the following pieces of context, extract the following information:

    authors: Who was this document authored or prepared by?\
    List the names of authors or answer Unknown if it cannot be determined.\
    Do not include people listed only under acknowledgements.

    summary: Give a summary of the content of this document,\
    including key takeaways, conclusions, and the purpose of the document.

    context: {context}

    {format_instructions}
    """
    prompt = ChatPromptTemplate.from_template(template=summary_template)

    docs = docsearch.similarity_search(summary_template, k=10)
    chain = load_qa_chain(OpenAI(openai_api_key=API_KEY), chain_type = "stuff", prompt=prompt)
    response = chain.run(input_documents=docs, format_instructions=format_instructions)
    # Handle weird json bug
    response = fix_langchain_json(response)
    try:
        output_dict = output_parser.parse(response)
        authors = output_dict["authors"]
        summary = output_dict["summary"]
        st.markdown('## Authors:')
        st.markdown(authors)
        st.markdown('## Summary:')
        st.markdown(summary)
        # summary_df = pd.DataFrame(output_dict, index=[0])
        # return st.table(summary_df)
    except:
        print(response)
        st.error("Cannot generate summary")
    return

def generate_response(docsearch, input_query, API_KEY):
    
    docs = docsearch.similarity_search(input_query, k=10)
    chain = load_qa_chain(OpenAI(openai_api_key=API_KEY), chain_type = "stuff")
    
    st.chat_message("user").write(input_query)
    with st.chat_message("assistant"):
        st.write("🧠 thinking...")
        if use_call_backs:
            st_callback = StreamlitCallbackHandler(st.container())
            response = chain.run(input_documents=docs, question=input_query, callbacks=[st_callback])
        else:
            response = chain.run(input_documents=docs, question=input_query)
        
        # Clean response
        response = response.replace('$', '')
        return(st.success(response))
    


# App layout ----------------------------------------------------------

# Input file loader
uploaded_file =  st.sidebar.file_uploader("Upload PDF file", type=['pdf'])


# Enter the API Key
API_KEY = st.sidebar.text_input('Enter your  OpenAI API Key', type='password', disabled=not(uploaded_file))
# Have API key AND uploaded file
if API_KEY.startswith("sk-") and (uploaded_file is not None):
    
    docsearch = load_pdf(uploaded_file, API_KEY)
    generate_summary(docsearch, API_KEY)

st.markdown('## Enter a query:')

# Some "fixed questions"
question_list = [
  'What was the purpose of this program?',
  'How successful was the program?',
  'Who are the partner organizations of this program?',
  'What issues was this program created to address?',
  'Who was this program created for?',
  'What were some challenges and limitations experienced during this program?',
  'Custom']
query_text = st.selectbox('Select an example query:', question_list, disabled=not (uploaded_file and API_KEY))


# Backend of app
if query_text == 'Custom':
    query_text = st.text_input('Enter your query', 
                               placeholder = 'Enter query here', 
                               disabled=not(uploaded_file and API_KEY))
    
if not API_KEY.startswith("sk-") and (uploaded_file is not None):
    st.warning('Please enter your OpenAI API key!', icon='⚠')


st.markdown('#')

st.markdown('## Response 🗣️')

# Use callback option
use_call_backs = tog.st_toggle_switch(label="Display callbacks", 
                    key="Display callbacks", 
                    default_value=False, 
                    label_after = False, 
                    inactive_color = '#D3D3D3', 
                    active_color="#11567f", 
                    track_color="#29B5E8"
                    )

generate_response(docsearch, query_text, API_KEY)

    
