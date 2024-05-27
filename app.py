import os
import streamlit as st
from dotenv import load_dotenv
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain_community.llms import LlamaCpp

from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Caching the model loader to avoid reloading on every rerun
@st.cache_resource
def load_model():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    cwd = os.getcwd()
    model_path = os.path.join(cwd, "model", "codellama-13b-instruct.Q4_K_M.gguf")
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=8000,
        n_gpu_layers=-1,
        n_batch=512,
        f16_kv=True,
        callback_manager=callback_manager,
        verbose=True,
    )
    return llm

# Caching the retriever loader to avoid reloading on every rerun
@st.cache_resource
def load_retriever(username='aajais'):
    cwd = os.getcwd()
    embd_model_path = os.path.join(cwd, "model", "nomic-embed-text-v1.5.Q5_K_S.gguf")
    embeddings = LlamaCppEmbeddings(model_path=embd_model_path, n_batch=512)
    db = DeepLake(dataset_path=f"hub://{username}/dipy-v2", read_only=True, embedding=embeddings)
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 70
    retriever.search_kwargs['k'] = 9
    return retriever

# Define the QA handling function
def handle_qa(user_input, llm, retriever):
    docs = retriever.invoke(user_input)
    # print(docs)
    prompt = PromptTemplate(
        template="""[INST]<<SYS>>Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.<</SYS>>
        {context}
        Question: {question}[/INST]
        Helpful Answer:""",
        input_variables=["question", "context"],
    )

    chain = prompt | llm | StrOutputParser()

    output = chain.invoke({"context": docs, "question": user_input})

    st.write(output)  # or st.text(output) for plain text

    # Optionally return the output if you need to use it elsewhere
    return output


# Initialize model and retriever outside of user interaction to leverage caching
llm = load_model()
retriever = load_retriever()

# Streamlit UI setup
st.title("Chat with a Bot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", "")
submit_button = st.button('Submit')

if submit_button and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    bot_response = handle_qa(user_input, llm, retriever)
    
    st.session_state.chat_history.append({"role": "bot", "content": bot_response})

for index, chat in enumerate(st.session_state.chat_history):
    role = "You" if chat["role"] == "user" else "Bot"
    unique_key = f"{role}_{index}"
    st.text_area(label=role, value=chat["content"], height=75, disabled=True, key=unique_key)
