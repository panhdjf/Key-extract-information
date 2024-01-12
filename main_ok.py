
# from langchain.document_loaders import PyPDFDirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import  HuggingFaceInstructEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# import torch
# from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# from langchain import HuggingFacePipeline, PromptTemplate
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import css, bot_template, user_template
# from os import listdir
# from os.path import isfile, join
# from transformers import pipeline
# from transformers import TextStreamer, pipeline

# import ipywidgets as widgets
# from IPython.display import display, HTML
# from sentence_transformers import SentenceTransformer
# import langchain
# langchain.verbose = True


# # device = torch.device("cpu")
# device = "cuda:0" if torch.cuda.is_available() else "cpu"

# from transformers import BitsAndBytesConfig

# nf4_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_quant_type="nf4",
#    bnb_4bit_use_double_quant=True,
#    bnb_4bit_compute_dtype=torch.bfloat16
# )

# # Read pdf
# loader = PyPDFDirectoryLoader("pdfs")
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
# texts = text_splitter.split_documents(docs)

# embeddings_vi_2 = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert", model_kwargs={"device": device})
# print('--------Embedding-------')

# vectorstore = FAISS.from_documents(texts, embedding=embeddings_vi_2)

# print('---------------Loading Model--------------')

# HUGGINGFACEHUB_API_TOKEN = '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'


# '''-----------------'''
# print('Loading model VNAI-llama2')

# model_name_or_path = "VietnamAIHub/Vietnamese_llama2_7B_8K_SFT_General_domain"
# cache_dir = "/home/vinbig/Documents/PA_Modeling/Retrieval_pdf/tmp"

# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True,cache_dir=cache_dir)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name_or_path,
#     # load_in_8bit=True,
#     torch_dtype=torch.float32,
#     pretraining_tp=1,
#     # use_auth_token=True,
#     # trust_remote_code=True,
#     cache_dir=cache_dir,
# )

# print('---------Loaded Model---------')



# SYSTEM_PROMPT = "Sử dụng các phần ngữ cảnh sau đây để trả lời câu hỏi ở cuối. Nếu không biết câu trả lời, bạn chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời."

# def generate_prompt(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
#     return f"""
# ### Câu hỏi:

# {system_prompt}
# {prompt}


# ### Trả lời:
# """.strip()

# streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# text_pipeline = pipeline(
#     "text-generation",
#     model=model.to('cpu'),
#     tokenizer=tokenizer,
#     max_new_tokens=1024,
#     temperature=0.,
#     top_p=0.95,
#     repetition_penalty=1.15,
#     streamer=streamer,
# )

# template = generate_prompt(
#     """
# {context}

# Câu hỏi: {question}
# """,
#     system_prompt=SYSTEM_PROMPT,
# )


# prompt = PromptTemplate(template=template, input_variables=["context", "question"])


# qa_chain = RetrievalQA.from_chain_type(
#     llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.}),
#     chain_type="stuff",
#     retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": prompt},
#     verbose=True,
# )

# import time

# while True: 
#     m = int(input("Nhập một số: ") )
#     if m == 0: 
#         break 
    
#     ques = input("Nhập câu hỏi: ")
#     start = time.time()
#     result = qa_chain(ques)
#     time_gen = time.time() - start
#     print("Time:", time_gen)




from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import  HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from langchain import HuggingFacePipeline, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from os import listdir
from os.path import isfile, join
from transformers import pipeline
from transformers import TextStreamer, pipeline

import ipywidgets as widgets
from IPython.display import display, HTML
from sentence_transformers import SentenceTransformer
import langchain
langchain.verbose = True
import time


# device = torch.device("cpu")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

from transformers import BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# Read pdf
loader = PyPDFDirectoryLoader("pdfs")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
texts = text_splitter.split_documents(docs)

#embedding metadata #GPU
embeddings_vi_2 = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert", model_kwargs={"device": device})
print('--------Embedding-------')

vectorstore = FAISS.from_documents(texts, embedding=embeddings_vi_2)

print('---------------Loading Model--------------')

HUGGINGFACEHUB_API_TOKEN = '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'

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

print('Loading model VNAI-llama2 - CPU')
cache_dir = '/home/vinbig/Documents/PA_Modeling/Prompt/bk_tmp'
tokenizer = AutoTokenizer.from_pretrained("bkai-foundation-models/vietnamese-llama2-7b-120GB", cache_dir=cache_dir)
# model = AutoModelForCausalLM.from_pretrained("bkai-foundation-models/vietnamese-llama2-7b-120GB", cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained("bkai-foundation-models/vietnamese-llama2-7b-120GB",cache_dir=cache_dir, quantization_config=nf4_config)


print('---------Loaded Model---------')

SYSTEM_PROMPT = "Sử dụng các phần ngữ cảnh sau đây để trả lời câu hỏi ở cuối. Nếu không biết câu trả lời, bạn chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời."

def generate_prompt(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    return f"""
### Câu hỏi:

{system_prompt}
{prompt}


### Trả lời:
""".strip()

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.2,
    top_p=0.95,
    repetition_penalty=1.15,
    streamer=streamer,
)

llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.2})

template = generate_prompt(
    """
{context}

Câu hỏi: {question}
""",
    system_prompt=SYSTEM_PROMPT,
)




prompt = PromptTemplate(template=template, input_variables=["context", "question"])


qa_chain = RetrievalQA.from_chain_type(
    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0}),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    verbose=True,
)

while True: 
    m = int(input("Nhập một số: ") )
    if m == 0: 
        break 
    
    ques = input("Nhập câu hỏi: ")
    start = time.time()
    # qa_chain.qa_chain.llm_chain.verbose=True
    result = qa_chain(ques)
    time_gen = time.time() - start
    print("Time:", time_gen)