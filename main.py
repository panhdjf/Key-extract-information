import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from htmlTemplates import css, bot_template, user_template

from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import torch
from transformers import  AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer, pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

import langchain
langchain.verbose = True
from PyPDF2 import PdfReader

from pdf2image import convert_from_bytes
from PIL import Image
from utils import get_OCR


# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = 'cpu'

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(docs):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(docs)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert", model_kwargs={"device": device})
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_text(ocr_res):
    text = ''
    phrases = ocr_res["phrases"]
    for phrase in phrases:
        text += phrase['text'] + '\n'
    return text

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def load_model():
    # model_name_or_path = "VietnamAIHub/Vietnamese_llama2_7B_8K_SFT_General_domain"
    # # model_name_or_path = "vilm/vinallama-7b"
    # cache_dir = '/home/vinbig/Documents/PA_Modeling/Retrieval_pdf/tmp1'
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name_or_path,
    #     torch_dtype=torch.float32,
    #     pretraining_tp=1,
    #     cache_dir=cache_dir,
    # )
    cache_dir = '/home/vinbig/Documents/PA_Modeling/Prompt/tmp'
    tokenizer = AutoTokenizer.from_pretrained("vilm/vinallama-7b", cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained("vilm/vinallama-7b", torch_dtype=torch.float32, cache_dir=cache_dir)

    return tokenizer, model


def template_prompt():
    SYSTEM_PROMPT = "Bạn là một Dược sĩ. Hãy phân tích giấy tờ y tế  sau để trả lời câu hỏi, trả về câu trả lời ở dạng JSON với các trường tương ứng trong câu hỏi."

    examples = [
    {
        "Câu hỏi": "Thông tin của bệnh nhân bao gồm {Họ tên bệnh nhân, Tuổi, Ngày sinh, Giới tính, Địa chỉ} là gì?",
        "Trả lời":
    """
    {
    'Tên':  string,
    'Tuổi':  string, 
    'Ngày sinh':  string,
    'Giới tính':  string, 
    'Địa chỉ':   string,
    }
    """
    },
    {
    "Câu hỏi": "Thông tin về đơn thuốc bao gồm {tên đơn vị/ cơ sở khám chữa phát hành thuốc, ngày ký giấy phát đơn thuốc, họ và tên bác sỹ kê và ký đơn, có chữ ký của bác sỹ không} là gì?",
    "Trả lời":
    """
    {
    'Tên đơn vị':  string,
    'Ngày ký':  string, 
    'Họ tên bác sỹ ':  string,
    'Chữ ký của bác sỹ':  bool, 
    }
    """
    }
    ]

    def generate_prompt(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        return f"""
    ### Câu hỏi:

    {system_prompt}
    {prompt}


    ### Trả lời:
    """.strip()

    template = generate_prompt( 
    """
    {context}

    Câu hỏi: {question}
    """,
        system_prompt=SYSTEM_PROMPT,
    )

    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    # prompt = FewShotPromptTemplate(examples=examples, example_prompt=example_prompt,suffix="Question: {user_question}", input_variables=["question"])

    return prompt

def get_conversation_chain(vectorstore):
    prompt = template_prompt()
    tokenizer, model = load_model()

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.95,
        repetition_penalty=1.15,
        streamer=streamer,
    )

    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.2, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=True,
        combine_docs_chain_kwargs=dict(prompt=prompt)

    )

    return conversation_chain

import time

def main():
    # load_dotenv()
    st.set_page_config(page_title="Extract infomation from prescription",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("AI medical chatbot :books:")

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_file = st.file_uploader(
            "Upload your PDF/Image here and click on 'Process'", accept_multiple_files=False)
        
    if uploaded_file:
        with st.spinner("Processing"):
            if 'pdf' in str(uploaded_file.type):
                images = convert_from_bytes(uploaded_file.read())
            else:
                images = [Image.open(uploaded_file) ]

            # get pdf text
            raw_text = ''
            for img in images:
                ocr_res = get_OCR(img, preprocess=False)
                text = get_text(ocr_res)
                raw_text += text

        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)

        st.image(images[0])

        # print(raw_text)
        # raw_text = get_pdf_text(pdf_docs)

        # get the text chunks
        text_chunks = get_text_chunks(raw_text)

        # create vector store
        vectorstore = get_vectorstore(text_chunks)
        
        # if st.button("Process"):
        with st.spinner("Processing"):
            # create conversation chain
            start = time.time()
            st.session_state.conversation = get_conversation_chain(
                vectorstore)
            print("Thời gian: ", time.time() - start)


if __name__ == '__main__':
    main()
















# from langchain.llms import HuggingFaceHub
# from langchain import PromptTemplate
# from langchain.chains import LLMChain
# import streamlit as st
# from langchain.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory

# from langchain import PromptTemplate

# from langchain.chains import SequentialChain
# import time

# import langchain
# langchain.verbose = True

# st.title = ('Extract infomation')

# display_text = st.text_input('Type a topic/person to get information')

# person_memory=ConversationBufferMemory(input_key='name', memory_key='chat_history')
# match_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')

# first_input_promt = PromptTemplate(
#     input_variables = ['name'],
#     template='Tell me something about celebrity {name}',
# )

# llm = HuggingFaceHub(model_kwargs={'temperature':0.5, 'max_length': 64}, repo_id='google/flan-t5-xxl')
# chain1 = LLMChain(llm=llm, prompt=first_input_promt, verbose=True, output_key='person', memory=person_memory)
# second_input_promt = PromptTemplate(
#     input_variables = ['person'],
#     template='Tell 3 major matches of {person} career'
# )

# chain2 = LLMChain(llm=llm, prompt=second_input_promt, verbose=True, output_key='matches', memory=match_memory)

# parent_chain = SequentialChain(
#     chains=[chain1, chain2], input_variables=['name'], output_variables=['person', 'matches'], verbose=True
# )

# if display_text:
#     st.write(parent_chain({'name':display_text}))

#     with st.expander('Person Name'):
#         st.info(person_memory.buffer)
    
#     with st.expander('Major matches'):
#         st.info(match_memory.buffer)

    