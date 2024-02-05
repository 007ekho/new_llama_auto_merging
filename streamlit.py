import os
import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
# from llama_index.node_parser import SentenceWindowNodeParser
from llama_index import load_index_from_storage
from llama_index import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.retrievers import AutoMergingRetriever
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.node_parser import HierarchicalNodeParser
from llama_index.node_parser import get_leaf_nodes


from llama_index import SimpleDirectoryReader

# from dotenv import load_dotenv, find_dotenv
# def get_openai_api_key():
#     _ = load_dotenv(find_dotenv())

#     return os.getenv("OPENAI_API_KEY")

# openai.api_key = get_openai_api_key()

openai.api_key = st.secrets.OPENAI_API_KEY




# Set the OpenAI API key


from llama_index import Document



llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1,system_prompt="Keep your answers technical and based on facts – do not hallucinate features.",api_key=openai.api_key)
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model="local:BAAI/bge-small-en-v1.5"
)



def build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index",
    chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context, service_context=merging_context
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=merging_context,
        )
    return automerging_index


def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=2,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine





st.set_page_config(page_title="Chat with the SOP DOCUMENTS, powered by LlamaIndex_aUTOMERGER", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
# openai.api_key = st.secrets.openai_key

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1,system_prompt="ALL ANSWERS MUST BE BASED ON THE CONTENT OF THE EXISTING DOCUMENT, IF THERE IS NO ANSWER FROM THE DOCUMENT OUTPUT: "PLEASE WITH AN OFFICER".Keep your answers technical and based on facts – do not hallucinate features.",api_key=openai.api_key)

st.title("Chat with the SOP docs, powered by LlamaIndex 💬🦙")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question "}
    ]

@st.cache_resource(show_spinner=False)

def load_data():
    try:
        with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
            documents = SimpleDirectoryReader(input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]).load_data()
            # documents =SimpleDirectoryReader(input_dir="./Data", recursive=True).load_data()
            # reader = SimpleDirectoryReader(input_dir="./Data", recursive=True)
            # documents = reader.load_data()
            
            
            llm = OpenAI(model="gpt-3.5-turbo", temperature=0, system_prompt="Keep your answers technical and based on facts – do not hallucinate features.", api_key=openai.api_key)
            
            automerging_index = build_automerging_index(documents, llm, embed_model="local:BAAI/bge-small-en-v1.5", save_dir="merging_index")
            
            return automerging_index
    except Exception as e:
        st.error(f"Error loading and indexing data: {e}")
        return None

automerging_index = load_data()

if automerging_index:
    if "chat_engine" not in st.session_state.keys():
        st.session_state.chat_engine = automerging_index.as_chat_engine(chat_mode="condense_question", verbose=True)

    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                user_input = st.session_state.messages[-1]["content"]
                
                # Query automerging_query_engine
                automerging_query_engine = get_automerging_query_engine(automerging_index)  # Replace with your actual function
                
                auto_merging_response = automerging_query_engine.query(user_input)
                st.write(str(auto_merging_response))

                # Generate response using chat_engine
                # response = st.session_state.chat_engine.chat(user_input)
                # st.write(response.response)

                # Add both responses to the message history
                message_auto_merging = {"role": "assistant", "content": str(auto_merging_response)}
                # message_chat = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message_auto_merging)
                # st.session_state.messages.append(message_chat)







# def load_data():
#     with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        
#         documents = SimpleDirectoryReader(input_files=["input_dir="./data", recursive=True"]).load_data()
        
#         # service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")
        
#         automerging_index = build_automerging_index(documents,llm,embed_model="local:BAAI/bge-small-en-v1.5",save_dir="merging_index"
# )
#         return automerging_index

# automerging_index = load_data()

# if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
#         st.session_state.chat_engine = automerging_index.as_chat_engine(chat_mode="condense_question", verbose=True)

# if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})

# for message in st.session_state.messages: # Display the prior chat messages
#     with st.chat_message(message["role"]):
#         st.write(message["content"])



# # If the last message is not from the assistant, generate a new response
# if st.session_state.messages[-1]["role"] != "assistant":
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             user_input = st.session_state.messages[-1]["content"]
            
#             # Query automerging_query_engine
#             automerging_query_engine = get_automerging_query_engine(automerging_index,)
            
#             auto_merging_response = automerging_query_engine.query(user_input)
#             st.write(str(auto_merging_response))

#             # Generate response using chat_engine
#             response = st.session_state.chat_engine.chat(user_input)
#             st.write(response.response)

#             # Add both responses to the message history
#             message_auto_merging = {"role": "assistant", "content": str(auto_merging_response)}
#             message_chat = {"role": "assistant", "content": response.response}
#             st.session_state.messages.append(message_auto_merging)
#             st.session_state.messages.append(message_chat)
