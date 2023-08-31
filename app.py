import streamlit as st

# import box
import timeit

# import yaml
from src.utils import setup_dbqa, get_config, read_markdown_file
from src.ui import set_png_as_page_bg


# Import config vars
cfg = get_config()

# Setup DBQA
dbqa = setup_dbqa(cfg)

st.set_page_config(
    page_title="Ask Canonical",
    page_icon="https://assets.ubuntu.com/v1/49a1a858-favicon-32x32.png",
    layout="wide",
)

set_png_as_page_bg("./media/chatbot-background-alpha0.3.png")

st.title(":penguin: Ask Canonical", anchor=False)

with st.sidebar:
    with st.form("model_params"):
        option = st.selectbox("Model", ["LLAMA_v2"])

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=cfg.TEMPERATURE,
        )
        batch_size = st.number_input("Batch size", value=cfg.MODEL_BATCH_SIZE)
        use_gpu = st.checkbox("Use GPU", value=cfg.USE_GPU)
        submit = st.form_submit_button(label="Reload model")

        if submit:
            cfg.TEMPERATURE = temperature
            cfg.MODEL_BATCH_SIZE = batch_size
            cfg.USE_GPU = use_gpu
            dbqa = setup_dbqa(cfg)
            print(cfg)
            st.success("Model was reloaded.")

    st.markdown(read_markdown_file("media/chat-documentation.md"))


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask your question!", key="chatbot_main_area"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})

    start = timeit.default_timer()
    response = dbqa({"query": question})
    end = timeit.default_timer()

    print(f'\nAnswer: {response["result"]}')
    print(f"Time to retrieve response: {end - start}")
    print("=" * 50)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response["result"])
    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response["result"]}
    )
