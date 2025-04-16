import streamlit as st
from modules import load_and_process_documents , create_vectorstore , save_faiss_to_bytes , load_faiss_from_bytes , chat_with_tools
from constants import KCS_VECTOR_STORE_PATH
from langchain_community.vectorstores import FAISS 
from langchain_openai import OpenAIEmbeddings  
from openai import OpenAI 
from dotenv import load_dotenv
load_dotenv()

client = OpenAI() 
def main() :
    st.title("Document Q&A with Google Search") 
    upload_option = st.radio("Upload Documents or Vectorstore or chat with KCS", ("Upload Documents",  "Upload Vectorstore" , "Chat with KCS")) 
    if upload_option == "Upload Documents":
            uploaded_files = st.file_uploader("Upload documents (PDF)", type=["pdf"], accept_multiple_files=True)
            if uploaded_files :
                  if st.button("Process documents") :
                        with st.spinner("Processing documents") :
                            texts_list = load_and_process_documents(uploaded_files) 
                            if texts_list :
                                vectorstore = create_vectorstore(texts_list)
                                st.session_state.vectorstore = vectorstore
                                st.success("Documents processed and vectorstore created!")
                                faiss_download = save_faiss_to_bytes(vectorstore)
                                if faiss_download:
                                    st.download_button("Download Vectorstore", data=faiss_download, file_name="faiss_index.pkl" , mime="application/octet-stream")
                            else:
                                st.warning("No valid documents found.")
    elif upload_option == "Upload Vectorstore":
        uploaded_faiss = st.file_uploader("Upload FAISS Vectorstore (.pkl)", type=["pkl"])
        if uploaded_faiss:
            if st.button("Load Vectorstore"):
                with st.spinner("Loading vectorstore..."):
                    faiss_bytes = uploaded_faiss.read()
                    vectorstore = load_faiss_from_bytes(faiss_bytes)
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.success("Vectorstore loaded!") 
                    else:
                        st.warning("Failed to load vectorstore.")
    elif upload_option == "Chat with KCS" :
        vectorstore = FAISS.load_local(KCS_VECTOR_STORE_PATH , OpenAIEmbeddings() , allow_dangerous_deserialization = True)
        if vectorstore :
            st.session_state.vectorstore = vectorstore
            st.success("Vectorstore loaded!")
        else :
            st.warning("Failed to load vectorstore.") 

    if "vectorstore" in st.session_state :
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "system", "content": "You are a helpful assistant. Use tools only when necessary.Prefer using the `retrieve_chunks` tool before considering others, especially for document-based questions. However if you don't find specific information related to question asked, always use `search_web' to find specific information before generating final answer"}
            ]
        query = st.text_input("Ask a question:")
        if query :
            with st.spinner("generating answer") :
                # print(completion.choices[0].message.tool_calls)
                # response = chat_with_tools(st.session_state.vectorstore , query)
                # st.write(response)
                response, updated_history = chat_with_tools(
                    st.session_state.vectorstore,
                    query,
                    st.session_state.chat_history
                )
                st.session_state.chat_history = updated_history  # Save updated history
                st.write(response.content)

if __name__ == "__main__":
    main()
             
            
             
         



                              
                              
