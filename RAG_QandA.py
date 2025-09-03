from langchain_huggingface import HuggingFaceEndpointEmbeddings
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import  PromptTemplate, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains.retrieval import create_retrieval_chain
import time

HF_API_KEYs = st.secrets.HF_API_KEY
GOOGLE_API_KEYs= st.secrets.GOOGLE_API_KEY
model_name = "gemini-1.5-flash-latest" #"gemini-1.5-pro-001"
if "kk" not in st.session_state:
    st.session_state.kk = ""
if "output_rag_ke" not in st.session_state:
    st.session_state.output_rag_key = ""
if "Chunk_Size" not in st.session_state:
    st.session_state.Chunk_Size = 700
if "Chunk_Overlap" not in st.session_state:
    st.session_state.Chunk_Overlap = 50
if "Ranking_k" not in st.session_state:
    st.session_state.Ranking_k = 5
if "Only_Code" not in st.session_state:
    st.session_state.Only_Code = False
if "Code_Documents" not in st.session_state:
    st.session_state.Code_Documents = False
if "tuning" not in st.session_state:
    st.session_state.tuning = False

def stream_data_rag():            
    for word in st.session_state.output_rag_key.split(" "):
        yield word + " "
        time.sleep(0.03)

@st.cache_data(show_spinner=False)
def pdf_extract(uploaded_file):
    if "pdf" in uploaded_file.type:
        docs = PdfReader(uploaded_file)
        document_pdf = docs.pages[:]

    pdf_text= ''
    for i in document_pdf:
        pdf_text = pdf_text + "\n" + (i.extract_text()) 
    return pdf_text

@st.cache_resource(show_spinner=False)
def Text_split_embedding_and_vectorstore(pdf_text,chunk_size_user=700,chunk_overlap_user=50):
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=chunk_size_user,
    chunk_overlap=chunk_overlap_user,
    length_function=len
    )
    chunks = text_splitter.split_text(pdf_text)
    embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2",huggingfacehub_api_token=HF_API_KEYs)
    db = FAISS.from_texts(chunks,embedding=embeddings)
    return db

@st.cache_data(show_spinner=False)
def only_document_prompt():
    chatprompt = PromptTemplate.from_template("""
simply answer the question given below in between triple grave accent ```
ignore the context given below in between triple hash ### 
                                      

                                                        
###
context: 
{context}
###
                                      


```
Question : {input}
```                                    
"""
)
    return chatprompt

@st.cache_data(show_spinner=False)
def only_code_prompt(code,user_prompt):
    only_code_prompting = f"""
You act as an AI assistant designed to answer user queries accurately and concisely based on the provided Code and Context. Use the provided Code and Context to derive your response, ensuring it is relevant and complete. Do not provide information beyond the context unless explicitly asked to do so.
<code>
{code}
</code>    
Question : {user_prompt}                                     
"""
    return only_code_prompting

@st.cache_resource(show_spinner=False)
def only_document_retriever(_chatprompt,_db,Ranking_k_user=5):
    llm = GoogleGenerativeAI(
    model = model_name,
    api_key = GOOGLE_API_KEYs
    )
    document_chain = create_stuff_documents_chain(llm,_chatprompt)
    retriever=_db.as_retriever(search_kwargs={'k': Ranking_k_user})
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    return retrieval_chain,retriever

@st.cache_resource(show_spinner=False)
def llm_model_code(only_code_prompt):
    llm = GoogleGenerativeAI(
    model=model_name,
    api_key=GOOGLE_API_KEYs
    )
    st.session_state.output_rag_key = llm.invoke(only_code_prompt)
    st.session_state.kk = only_code_prompt
    
def only_document_answer(_retrievalchain,_userprompt):
    Answer = _retrievalchain.invoke({"input":_userprompt})
    st.session_state.output_rag_key = Answer["answer"]
    st.session_state.kk = Answer

title_conatiner= st.container()
title_conatiner.title("👨‍💻 RAG Q&A")
title_conatiner.caption("Decode with Confidence: RAG Q&A at Your Fingertips")
buttons = st.container(border=True)
user_prompt = st.chat_input("Write your Query")
col1,col2,col3 = buttons.columns(3,gap='large')
if  col1.toggle("Only Code",key ="Only_Code"):
    code_snippets = buttons.text_area("Enter Your Code Here",height=120,help="Only applicable for Python",placeholder=f"a = 5\nb = 5\nc = a + b\nprint(c)")
    if st.session_state.tuning or st.session_state.Only_Documents:
        st.session_state.Only_Documents = False
        st.session_state.tuning = False
    if user_prompt:
        only_code_prompting = only_code_prompt(code_snippets,user_prompt)
        llm_model_code(only_code_prompting)
        # st.write(only_code_prompting)

if col2.toggle("Only Documents",key ="Only_Documents"):
    file = buttons.file_uploader("Upload Here",type=["pdf"],help="Only PDF Document",accept_multiple_files=False)
    if st.session_state.Only_Code:
        st.session_state.Only_Code = False

    if user_prompt:
        chatprompt = only_document_prompt()
        if st.session_state.tuning:
            db= Text_split_embedding_and_vectorstore(pdf_extract(file),chunk_size_user=st.session_state.Chunk_Size,chunk_overlap_user=st.session_state.Chunk_Overlap)
            retriver_chain,retr_cha = only_document_retriever(chatprompt,db,Ranking_k_user=st.session_state.Ranking_k)
            st.write(retr_cha)
        else:
            db= Text_split_embedding_and_vectorstore(pdf_extract(file))
            retriver_chain = only_document_retriever(chatprompt,db)
            retriver_chain,retr_cha = only_document_answer(retriver_chain,user_prompt)
            st.write(retr_cha)
        # st.write(st.session_state.kk)

if col3.toggle("Tuning",key="tuning"):
        buttons.divider()
        buttons.write(f"Tuning")
        colum1,colum2 = buttons.columns(2)
        colum3,colum4 =buttons.columns(2)
        st.session_state.Chunk_Size = colum1.number_input("Chunk Size",value=st.session_state.Chunk_Size)
        st.session_state.Chunk_Overlap = colum2.number_input("Chunk Overlap",value=st.session_state.Chunk_Overlap)
        st.session_state.Ranking_k = colum3.number_input("Ranking",value=st.session_state.Ranking_k)
        colum4.code(f"""Chunk_Size : {st.session_state.Chunk_Size},
Chunk_Overlap : {st.session_state.Chunk_Overlap},
Ranking_k : {st.session_state.Ranking_k}""")
    
chat = st.container(border=True)
if user_prompt and (st.session_state.Only_Code or st.session_state.Only_Documents):
    with chat.empty():
        with st.chat_message("user"):
            st.markdown(user_prompt)
        with chat.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.write_stream(stream_data_rag)
