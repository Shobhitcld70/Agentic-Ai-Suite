import streamlit as st
import sys, os
import uuid
from langchain_core.messages import HumanMessage

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from langgraph_backend import chatbot

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id, "New Chat")
    st.session_state['message_history'] = [] 

def add_thread(thread_id, name):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'][thread_id] = name

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
    
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
    
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = {}
    
add_thread(st.session_state['thread_id'], "Current Chat")

st.sidebar.title('Langgraph Chatbot')

if st.sidebar.button('New Chat', key='new_chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id, name in st.session_state['chat_threads'].items():
    if st.sidebar.button(name, key=thread_id):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)
        
        temp_messages = []
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})
            
        st.session_state['message_history'] = temp_messages

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here: ')

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)
        
    config = {'configurable': {'thread_id': st.session_state['thread_id']}}

    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            chunk.content for chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=config,
                stream_mode='messages'
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

    # Update thread name if it's still generic
    if st.session_state['chat_threads'][st.session_state['thread_id']] in ["New Chat", "Current Chat"]:
        st.session_state['chat_threads'][st.session_state['thread_id']] = user_input[:30] + ("..." if len(user_input) > 30 else "")
