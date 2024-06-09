import pathlib
import textwrap
import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown
import PIL.Image
import os

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig, pipeline
from huggingface_hub import login

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

import streamlit as st
from PIL import Image
import io
import re

genai.configure(api_key="GOOGLE_API_KEY")
hf_token = "HUGGING_FACE_API_KEY"
hf_token = login(hf_token)

gemini_prompt =  """
Please analyze the given graph in detail. 

Your explanation should include:
- The title and context of the graph.
- A description of the x-axis and y-axis, including their labels and units.
- The key trends, patterns, and significant data points.
- Any anomalies or outliers present in the graph.
- A summary of what the graph indicates and its implications.

Ensure the explanation is comprehensive, between 800-1200 words, and includes specific examples and data points from the graph to support your analysis.

"""

#Multi model(text-images-audio) gemini-1.5-flash - Using it for Image to text
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

#llama2 chat hf Quantized Model
model_id = "meta-llama/Llama-2-7b-chat-hf"

quantization_config = QuantoConfig(load_in_8bit_fp32_cpu_offload=True)

llma2_model = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map="auto",
                                             quantization_config=quantization_config)

llma2_tokenizer = AutoTokenizer.from_pretrained(model_id)

#sentence-transformer embedding
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


pipe = pipeline("text-generation", model=llma2_model, tokenizer=llma2_tokenizer, max_new_tokens=250)
hf = HuggingFacePipeline(pipeline=pipe)

def extract_answer(text):
    match = re.search(r'Answer:(.*?)(?=Explanation:|$)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return "The answer is not available in the context."

def chatbot_response(user_question):
    prompt_template = """
    Answer the question as thoroughly as possible based on the provided context.
    Ensure to include all relevant details.
    If the answer cannot be determined from the provided context,
    simply state, "The answer is not available in the context."
    Avoid providing incorrect answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(hf, chain_type="stuff", prompt=prompt)

    vector_store = st.session_state['vector_store']
    docs = vector_store.similarity_search(user_question)
    response = chain(
            {"input_documents":docs, "question": user_question}
            , return_only_outputs=True)
    ans = extract_answer(response["output_text"])
    return f"{ans}"

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def handle_user_input():
    user_input = st.session_state.user_input
    if user_input:
        response = chatbot_response(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Assistant", response))
        st.session_state.user_input = ""

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

with st.sidebar:
    uploaded_files = st.file_uploader("Choose Images", accept_multiple_files=True)

    if st.button("Text Summarization"):
        st.session_state['task'] = 'summarization'

    if st.button("Question and Answer"):
        st.session_state['task'] = 'qa'

if 'task' in st.session_state and st.session_state['task'] == 'summarization':
    if 'geminiResponse' in st.session_state:
        text_res = st.session_state['geminiResponse']
        st.write(text_res)

if uploaded_files:
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        image = Image.open(io.BytesIO(bytes_data))
        #gemini response
        response = gemini_model.generate_content([gemini_prompt, image], stream=True)
        response.resolve()
        st.session_state['geminiResponse'] = response.text

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
        text_chunks = text_splitter.split_text(st.session_state['geminiResponse'])
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        st.session_state['vector_store'] = vector_store

        st.write("DONE")

        st.write("Filename:", uploaded_file.name)
        st.image(image, caption=uploaded_file.name, use_column_width=True)

if 'task' in st.session_state and st.session_state['task'] == 'qa':
    st.markdown("---")
    chatbot_placeholder = st.empty()

    with chatbot_placeholder.container():
        st.write("### Chatbot")
        for sender, message in st.session_state.chat_history:
            with st.chat_message(sender.lower()):
                st.write(message)

        user_input = st.text_input("You: ", key="user_input", on_change=handle_user_input)

        if user_input:
            with st.chat_message("user"):
                st.write(user_input)

            response = chatbot_response(user_input)
            with st.chat_message("assistant"):
                st.write(response)
