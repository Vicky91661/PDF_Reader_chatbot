### RAG Q&A Conversational With PDF Including Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessagesHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_groq import CHatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loader import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory


import os
from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name = "allMiniLM-L6-v2")


## Set up Streamlit
st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload PDF's and chat with their content")


## Input the Groq API Key
api_key = st.text_input("Enter your Groq API Key:",type="password")

## check if groq api key is provided

if api_key:
    llm = ChatGroq(groq_api_key = api_key, model_name = "Gemma2-9b-It")
    ## Chat Interface
    session_id = st.text_input("Session ID",value="default")

    ## Statefully manage chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}

    upload_files = st.file_uploader("Choose a PDF file", type="pdf",accept_multiple_files = False)

    ## Process uploaded PDF's
    if upload_files:
        documents = []
        for uploaded_file in upload_files:
            temppdf = f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = upload_files.name
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)
        # Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap = 200)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents = splits, embeddings = embeddings)
        retriever = vectorstore.as_retriever()


        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "Which might reference context in the chat history, "
            "formulate a standlone question which can be understood"
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        
        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        ## Answer question
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the ?? fsfddsf fgy dfgvcvb xb  vbbvbnh bv  cvb vb vbvbvbc  vbv bv cvbv vcvvvb vbc cvbc bbnbn bn bnbnbbn bvn "
            "answer concise."
            "\n\n"
            "{context}"
        )


        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )
        # 
        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session_id:str) -> BaseChatMessagesHistory :
            if session_id not in st.session_state.store:
                st.session_state[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        # conversatioanl rag chain 
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_message_key ="input",
            history_message_key = "chathistory",
            output_message_key = "answer"
        ) 
        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input":user_input},
                config = {
                    "configurable": {"session_id":session_id}

                }
            )
            st.write(st.session_state.store)
            st.write("Assistant:",response['answer'])
            st.write("chat history:",session_history.messages)
        else:
            st.warning("Please enter the Groq API Key")
    
