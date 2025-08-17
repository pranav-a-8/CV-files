from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from datetime import datetime
from langchain_ollama.llms import OllamaLLM

import streamlit as st
import json
import os
import time





@st.cache_resource
def get_local_model():
    return OllamaLLM(model="deepseek-r1:8b")
llm = get_local_model()

def load_documents():

    loader = DirectoryLoader(os.getcwd())
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    return docs

@st.cache_resource
def get_chroma_instance():
    # Get the documents split into chunks
    docs = load_documents()

    # create the open-sourc e embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # load it into Chroma
    return Chroma.from_documents(docs, embedding_function)

db = get_chroma_instance()  

def query_documents(question):
    """
    Uses RAG to query documents for information to answer a question

    Example call:

    query_documents("What are the action items from the meeting on the 20th?")
    Args:
        question (str): The question the user asked that might be answerable from the searchable documents
    Returns:
        str: The list of texts (and their sources) that matched with the question the closest using RAG
    """
    similar_docs = db.similarity_search(question, k=5)
    docs_formatted = list(map(lambda doc: f"Source: {doc.metadata.get('source', 'NA')}\nContent: {doc.page_content}", similar_docs))

    return docs_formatted   

def prompt_ai(messages):
    # Fetch the relevant documents for the query
    user_prompt = messages[-1].content
    retrieved_context = query_documents(user_prompt)
    formatted_prompt = f"Context for answering the question:\n{retrieved_context}\nQuestion/user input:\n{user_prompt}"    

    # Prompt the AI with the latest user message
    llm = get_local_model()
    full_response = llm.invoke(formatted_prompt)

    # Separate the thinking process from the final response
    start_index = full_response.find("<think>")
    end_index = full_response.find("</think>")

    if start_index != -1 and end_index != -1:
        thinking_process = full_response[start_index + 7:end_index]
        final_response = full_response[end_index + 8:]
    else:
        thinking_process = ""
        final_response = full_response

    return thinking_process, final_response

def stream_text(text, container):
    streamed_text = ""
    for char in text:
        streamed_text += char
        container.write(streamed_text)
        time.sleep(0.02)  # Adjust the delay as needed

def main():
    st.title("Chat with Local Documents")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=f"You are a personal assistant who answers questions based on the context provided if the provided context can answer the question. You only provide the answer to the question/user input and nothing else. The current date is: {datetime.now().date()}")
        ]    

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        message_json = json.loads(message.model_dump_json())
        message_type = message_json["type"]
        if message_type in ["human", "ai", "system"]:
            with st.chat_message(message_type):
                st.markdown(message_json["content"])        

    # React to user input
    if prompt := st.chat_input("What questions do you have?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append(HumanMessage(content=prompt))

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            thinking_process, final_response = prompt_ai(st.session_state.messages)

            if thinking_process:
                with st.expander("Thinking Process"):
                    thinking_container = st.empty()
                    stream_text(thinking_process, thinking_container)

            response_container = st.empty()
            stream_text(final_response, response_container)
        
        # Add AI response to chat history as a string
        st.session_state.messages.append(AIMessage(content=final_response))

if __name__ == "__main__":
    main()
