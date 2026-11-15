import os
from context import TWIN_SYSTEM_PROMPT
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
from tools import handle_tool_calls, tools

load_dotenv(override=True)

# Use your working 20B model via Groq
MODEL_NAME = "openai/gpt-oss-20b"

# Initialize Groq client securely for local .env or cloud secrets
api_key = os.getenv("GROQ_API_KEY")
if not api_key and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]

openai = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

st.set_page_config(page_title="Digital Twin", page_icon="🤖")
st.title("🤖 Digital Twin")
st.caption("Talk to my AI twin about my career")

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": TWIN_SYSTEM_PROMPT}
    ]

# Display prior chat messages (ignoring system prompt)
for msg in st.session_state.messages:
    if msg["role"] in ("user", "assistant") and msg.get("content"):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Handle user input via Streamlit chat box
if prompt := st.chat_input("Ask me anything about my experience or projects..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=st.session_state.messages,
            tools=tools,
        )
        
        # Handle tool calls if the model invokes them
        while response.choices[0].finish_reason == "tool_calls":
            msg_obj = response.choices[0].message
            results = handle_tool_calls(msg_obj.tool_calls)
            st.session_state.messages.append(msg_obj)
            st.session_state.messages.extend(results)
            response = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=st.session_state.messages,
                tools=tools,
            )

        reply = response.choices[0].message.content
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
