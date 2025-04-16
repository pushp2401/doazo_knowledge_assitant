import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain_openai import OpenAIEmbeddings  
from langchain.tools import Tool
# from langchain.utilities import SerpAPIWrapper
from langchain_community.utilities import SerpAPIWrapper
from openai import OpenAI 
import tempfile
import pickle
import os
# from langchain.vectorstores import FAISS

import json

import pickle
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI() 


def load_and_process_documents(uploaded_files):
    """Loads and processes documents from uploaded files."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    all_texts = []

    for uploaded_file in uploaded_files:
        temp_dir = tempfile.TemporaryDirectory()
        temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(temp_filepath)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}")
                continue

            documents = loader.load()
            texts = text_splitter.split_documents(documents)
            all_texts.extend(texts)

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

        temp_dir.cleanup()

    return all_texts

def create_vectorstore(text_list) :
    faiss_index = FAISS.from_documents(text_list, OpenAIEmbeddings())
    return faiss_index

def save_faiss_to_bytes(vectorstore) -> bytes:
    faiss_folder = "temp_faiss"
    if not os.path.exists(faiss_folder):
        os.makedirs(faiss_folder)

    # Save the FAISS index and metadata
    vectorstore.save_local(faiss_folder)

    # Bundle all parts into a single bytes object (pickle)
    with open(os.path.join(faiss_folder, "index.faiss"), "rb") as f:
        faiss_data = f.read()
    with open(os.path.join(faiss_folder, "index.pkl"), "rb") as f:
        metadata = f.read()

    # Clean up temp files (optional)
    os.remove(os.path.join(faiss_folder, "index.faiss"))
    os.remove(os.path.join(faiss_folder, "index.pkl"))
    os.rmdir(faiss_folder)

    # Combine both files into one pickle
    return pickle.dumps({
        "faiss_data": faiss_data,
        "metadata": metadata
    })

def load_faiss_from_bytes(faiss_bytes):
    
    # Deserialize the combined pickle (safe because you created it)
    data = pickle.loads(faiss_bytes)

    with tempfile.TemporaryDirectory() as faiss_folder:
        # Write files back to temp directory
        with open(os.path.join(faiss_folder, "index.faiss"), "wb") as f:
            f.write(data["faiss_data"])
        with open(os.path.join(faiss_folder, "index.pkl"), "wb") as f:
            f.write(data["metadata"])

        # Load FAISS vectorstore with allow_dangerous_deserialization=True
        vectorstore = FAISS.load_local(
            faiss_folder,
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True
        )

    return vectorstore
def retrieve_chunks(vectorstore , query: str) -> str:
    docs = vectorstore.similarity_search(query, k=4)
    return "\n\n".join([doc.page_content for doc in docs])

search = SerpAPIWrapper()
def search_web(query: str) -> str:
    return search.run(query)


tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_chunks",
            "description": "Retrieve relevant information from the document vectorstore. ",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's question or search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the internet for current or general knowledge",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search on the web"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# def chat_with_tools(vs , user_query: str):
#     # import pdb; pdb.set_trace()
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant. Use tools only when necessary."},
#         {"role": "user", "content": user_query}
#     ]

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=messages,
#         tools=tool_definitions,
#         tool_choice="auto"  # This lets the model decide
#     )

#     message = response.choices[0].message

#     # If a tool is used
#     # if "tool_calls" in message:
#     if message.tool_calls :
#         results = []
#         for tool_call in message.tool_calls:
#             name = tool_call.function.name
#             arguments = json.loads(tool_call.function.arguments)

#             if name == "retrieve_chunks":
#                 result = retrieve_chunks(vs , **arguments)
#             elif name == "search_web":
#                 result = search_web(**arguments)

#             # Append tool result
#             messages.append(message)  # The tool call message
#             messages.append({
#                 "role": "tool",
#                 "tool_call_id": tool_call.id,
#                 "name": name,
#                 "content": result
#             })

#         # Ask GPT again with tool outputs
#         final = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=messages
#         )
#         # return final["choices"][0]["message"]["content"]
#         return final

#     # If no tool is used
#     else : 
#         return message

def chat_with_tools(vs, user_query: str, messages=None):
    if messages is None:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use tools only when necessary. Prefer using the `retrieve_chunks` tool before considering others, especially for document-based questions. However if you don't find specific information related to question asked, always use `search_web' to find specific information before generating final answer"}
        ]

    # Add the new user query to chat history
    messages.append({"role": "user", "content": user_query})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tool_definitions,
        tool_choice="auto"  # Let the model decide
    )

    message = response.choices[0].message

    if message.tool_calls:
        results = []
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            # Call the appropriate tool
            if name == "retrieve_chunks":
                result = retrieve_chunks(vs, **arguments)
            elif name == "search_web":
                result = search_web(**arguments)
            else:
                result = f"Tool '{name}' not implemented."

            # Append tool call and tool response to chat history
            messages.append(message)  # The tool call message
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": result
            })

        # Get model response with tool outputs
        final = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        final_message = final.choices[0].message
        messages.append(final_message)

        return final_message, messages

    else:
        messages.append(message)
        return message, messages
