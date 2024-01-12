# from typing import List, Optional
# from langchain import HuggingFacePipeline
# from kor.nodes import Object, Text, Number
# from langchain.prompts import PromptTemplate
# from langchain.output_parsers import StructuredOutputParser, OutputFixingParser
# from langchain.chains.openai_functions.extraction import create_extraction_chain_pydantic
# from langchain.prompts.few_shot import FewShotPromptTemplate


# import pandas as pd
# from pydantic import BaseModel, Field, validator
# from kor import extract_from_documents, from_pydantic, create_extraction_chain

# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import torch
# from transformers import  AutoModelForCausalLM, AutoTokenizer, AutoModel
# from transformers import TextStreamer, pipeline
# import os
# import json
# os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
# import asyncio
# from transformers import AutoConfig, BitsAndBytesConfig
# from langchain.output_parsers import PydanticOutputParser

# from pdf2image import convert_from_bytes
# from PIL import Image
# from utils import get_OCR

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# # device = torch.device("cpu")

# '''----LOAD MODEL----'''
# nf4_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_quant_type="nf4",
#    bnb_4bit_use_double_quant=True,
#    bnb_4bit_compute_dtype=torch.bfloat16
# )
# def load_model():
#     cache_dir = '/home/vinbig/Documents/PA_Modeling/Prompt/bk_tmp'
#     tokenizer = AutoTokenizer.from_pretrained("bkai-foundation-models/vietnamese-llama2-7b-120GB", cache_dir=cache_dir)
#     model = AutoModelForCausalLM.from_pretrained("bkai-foundation-models/vietnamese-llama2-7b-120GB", cache_dir=cache_dir)
#     # model = AutoModelForCausalLM.from_pretrained("bkai-foundation-models/vietnamese-llama2-7b-120GB",cache_dir=cache_dir, quantization_config=nf4_config)
#     streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

#     text_pipeline = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=512,
#         temperature=0.01,
#         top_p=0.95,
#         repetition_penalty=1.15,
#         streamer=streamer,
#     )

#     llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.01, "max_length":512, "stop": "</exp>"})
#     return llm



# def get_label(path):
#     with open(path, 'r', encoding='utf-8') as file:
#         data = json.load(file)

#     # Các trường cần trích xuất
#     fields = [
#         'current institute',
#         'name',
#         'gender',
#         'birth',
#         'address',
#         'id bhyt',
#         'diagnosis',
#         'date in',
#         'doctor name'
#     ]

#     # Tạo dictionary kết quả
#     result = {}
#     for field in fields:
#         if field in data:
#             result[field] = data[field]['value']

#     return result

# def load_conversation(filename):

#     with open(filename, 'r') as f:
#         conversation = f.read()

#     return conversation

# def load_doc_label(path_txts, path_jsons):
#     name_txts = os.listdir(path_txts)
#     docs = []
#     labels = []
#     for name in name_txts:
#         doc = load_conversation(os.path.join(path_txts, name))

#         n = name.split('.')[0] + '.json'
#         path = os.path.join(path_jsons, n)
#         label = get_label(path)
#         docs.append(doc)
#         labels.append(label)
#     return docs, labels
        
# def get_example(docs, labels):
#     examples = []
#     for i in range(len(docs)):
#         ques = {"question": docs[i]}
#         ans = {"answer": "<exp>" + str(labels[i])[1:-1] + "</exp>"}
#         ques.update(ans)
#         examples.append(ques)
#     return examples

# def get_text(ocr_res):
#     text = ''
#     phrases = ocr_res["phrases"]
#     for phrase in phrases:
#         text += phrase['text'] + '\n'
#     return text

# def read_imgs(path_file):
#     images = [Image.open(path_file)]

#     # get pdf text
#     raw_text = ''
#     for img in images:
#         ocr_res = get_OCR(img, preprocess=False)
#         text = get_text(ocr_res)
#         raw_text += text
#     return raw_text

# def get_text_chunks(text):
#     doc = Document(page_content=text)
#     split_docs = RecursiveCharacterTextSplitter().split_documents([doc])
#     return split_docs


# path_txts = '/home/vinbig/Documents/PA_Modeling/Prompt/private_test_Pharma/ocr_text'
# path_jsons = '/home/vinbig/Documents/PA_Modeling/Prompt/prescription_label_text/KIEs'
# docs, labels = load_doc_label(path_txts, path_jsons)
# examples = get_example(docs, labels)
# examples = examples[:20]

# SYSTEM_PROMPT = '''Bạn là chuyên gia trong lĩnh vực y tế. Mục tiêu của bạn là cung cấp cho người dùng thông tin được trích xuất từ kiến ​​thức được cung cấp. Hãy suy nghĩ từng bước một và đừng bao giờ bỏ qua bước nào.
#     Dưới đây là những thông tin bạn cần trích xuất:
#     - current_institute: Tên bệnh viện/ Tên phòng khám phát hành đơn thuốc
#     - name: Tên bệnh nhân
#     - gender: Giới tính của bệnh nhân
#     - birth: (ngày, tháng) năm sinh bệnh nhân
#     - age: Tuổi bệnh nhân
#     - address: Địa chỉ bệnh nhân
#     - tel_customer: Số điện thoại của bệnh nhân
#     - id_bhyt: Số thẻ bảo hiểm y tế của bệnh nhân
#     - diagnosis: Chẩn đoán bệnh
#     - date_in: Ngày phát hành đơn thuốc
#     - doctor_name: Họ tên Bác sĩ kê đơn
# '''

# def generate_prompt(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
#     return f"""
# ### Câu hỏi:

# {system_prompt}
# {prompt}


# ### Trả lời:
# """.strip()

# template = generate_prompt(
#     """
# Câu hỏi: {question}

# Trả lời: {answer}
# """,
#     system_prompt=SYSTEM_PROMPT,
# )




# example_prompt = PromptTemplate(template=template, input_variables=["question", "answer"])



# # example_prompt = PromptTemplate(
# #     template='''Bạn là chuyên gia trong lĩnh vực y tế. Mục tiêu của bạn là cung cấp cho người dùng thông tin được trích xuất từ kiến ​​thức được cung cấp. Hãy suy nghĩ từng bước một và đừng bao giờ bỏ qua bước nào.
# #     Dưới đây là những thông tin bạn cần trích xuất:
# #     - current_institute: Tên bệnh viện/ Tên phòng khám phát hành đơn thuốc
# #     - name: Tên bệnh nhân
# #     - gender: Giới tính của bệnh nhân
# #     - birth: (ngày, tháng) năm sinh bệnh nhân
# #     - age: Tuổi bệnh nhân
# #     - address: Địa chỉ bệnh nhân
# #     - tel_customer: Số điện thoại của bệnh nhân
# #     - id_bhyt: Số thẻ bảo hiểm y tế của bệnh nhân
# #     - diagnosis: Chẩn đoán bệnh
# #     - date_in: Ngày phát hành đơn thuốc
# #     - doctor_name: Họ tên Bác sĩ kê đơn

# #     Thông tin: {question}

# #     Kết quả: {answer}
# #     ''',
# #     input_variables=["question", "answer"],
# # )

# prompt = FewShotPromptTemplate(
#     examples=examples,
#     example_prompt=example_prompt,
#     suffix="Câu hỏi: {inputText}",
#     input_variables=["inputText"], 
# )



# path_input = '/home/vinbig/Documents/PA_Modeling/Prompt/Long_Chau_28.txt'
# input = load_conversation(path_input)
# split_docs = get_text_chunks(input)

# input_text = prompt.format(inputText = split_docs[0].page_content)

# llm = load_model()
# llm(input_text)




'''------------------------------------------------------'''



# from langchain.prompts.few_shot import FewShotPromptTemplate
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# hf_token = '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'
# from torch import cuda, bfloat16
# import transformers
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import RetrievalQA
# from langchain import HuggingFacePipeline, PromptTemplate
# from torch import cuda, bfloat16
# import transformers
# from transformers import  AutoModelForCausalLM, AutoTokenizer
# from transformers import TextStreamer, pipeline
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import ConversationalRetrievalChain
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.chains import LLMChain
# from langchain.schema import StrOutputParser
 
# from pydantic import BaseModel
 
 
# class Item(BaseModel):
#     content: str
# class Prompt(BaseModel):
#     question: str

# bnb_config = transformers.BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type='nf4',
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=bfloat16
# )
 
# class LLM:
#     def __init__(self, filename):
#         self.model_embeddings = "keepitreal/vietnamese-sbert"
#         self.cache_dir = '/home/vinbig/Documents/PA_Modeling/Prompt/bk_tmp'
#         self.model_name_or_path = "bkai-foundation-models/vietnamese-llama2-7b-120GB"
#         self.model_kwargs = {'device': 'cuda:0'}
#         self.filename = filename
 
 
#         self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path,
#                                              trust_remote_code=True,
#                                              quantization_config=bnb_config,
#                                              token = hf_token,
#                                              cache_dir = self.cache_dir)
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True, model_kwargs=self.model_kwargs, token = hf_token, cache_dir = self.cache_dir)
#         self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
#         self.text_pipeline = pipeline(
#             "text-generation",
#             model=self.model,
#             tokenizer=self.tokenizer,
#             max_new_tokens=120,
#             temperature=0.1,
#             top_p=0.95,
#             repetition_penalty=1.15,
#             streamer=self.streamer,
#         )
#         self.llm = HuggingFacePipeline(pipeline = self.text_pipeline, model_kwargs={"temperature": 0.1, "max_length":512,'device': 'cuda:0'})
#         self.embeddings = HuggingFaceEmbeddings(model_name = self.model_embeddings,  model_kwargs = self.model_kwargs)
#         # self.db = FAISS.load_local('/home/huy.nguyen/langchain_template/dbyte', self.embeddings)
#         # self.parser = PydanticOutputParser(pydantic_object=Patient)
#         self.prompt_template = """You are an expert in medical field. Your goals is to provide user useful answer from provided knowledge. Think step by step and never ignore any step.
#                         Remember:
#                         - always answer in Vietnamese.
#                         - dont try to generate other answers and questions.
#                         - to be honest if you don't know, don't try to answer.
#                         knowledge : {context}
 
 
#                         question : {question}
 
#                         """
#         self.prompt = PromptTemplate(template=self.prompt_template, input_variables=['context', 'question'])
#         self.context = self.load_conversation(filename)
#         self.db = self.create_db(self.context)
        
 
#     def load_conversation(self, filename):

#         with open(filename, 'r') as f:
#             context = f.read()

#         return context

#     def create_db(self,context):
#         text_splitter = CharacterTextSplitter(
#             separator="\n",
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len
#         )
#         docs = text_splitter.split_text(context)
 
#         self.db = FAISS.from_texts(docs, self.embeddings)
#         return self.db
#     def format_docs(self,docs):
#         return "\n\n".join(doc.page_content for doc in docs)
#     def result(self, question):
#         rag_chain = (
#             {"context": self.db.as_retriever(search_kwargs={"k": 2}) | self.format_docs, "question": RunnablePassthrough()}
#             | self.prompt
#             | self.llm.bind(stop =["\n"])
#             | StrOutputParser())
#         return rag_chain.invoke(question)

# # context = Item().content


# filename = '/home/vinbig/Documents/PA_Modeling/Prompt/private_test_Pharma/ocr_text/Long_Chau_46.txt'
# chatbot = LLM(filename)

# # docs = chatbot.create_db(context)[0]
# # input_text = prompt.format(inputText = docs)


# # prompt = Prompt()
# ques = input("Nhập câu hỏi: ")
# result = chatbot.result(ques)
 
 
'''------------------------------------------------------'''
 
from dotenv import load_dotenv
# from pytesseract import image_to_string
from PIL import Image
from io import BytesIO
# import pypdfium2 as pdfium
import streamlit as st
# import multiprocessing
# from tempfile import NamedTemporaryFile
import pandas as pd
import json
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
# from tempfile import NamedTemporaryFile
from jsonformer.format import highlight_values
from jsonformer.main import Jsonformer
import time
load_dotenv()

# 1. Convert PDF file into images via pypdfium2


# def convert_pdf_to_images(file_path, scale=300/72):

#     pdf_file = pdfium.PdfDocument(file_path)

#     page_indices = [i for i in range(len(pdf_file))]

#     renderer = pdf_file.render(
#         pdfium.PdfBitmap.to_pil,
#         page_indices=page_indices,
#         scale=scale,
#     )

#     final_images = []

#     for i, image in zip(page_indices, renderer):

#         image_byte_array = BytesIO()
#         image.save(image_byte_array, format='jpeg', optimize=True)
#         image_byte_array = image_byte_array.getvalue()
#         final_images.append(dict({i: image_byte_array}))

#     return final_images

# # 2. Extract text from images via pytesseract


# def extract_text_from_img(list_dict_final_images):

#     image_list = [list(data.values())[0] for data in list_dict_final_images]
#     image_content = []

#     for index, image_bytes in enumerate(image_list):

#         image = Image.open(BytesIO(image_bytes))
#         raw_text = str(image_to_string(image))
#         image_content.append(raw_text)

#     return "\n".join(image_content)


# def extract_content_from_url(url: str):
#     images_list = convert_pdf_to_images(url)
#     text_with_pytesseract = extract_text_from_img(images_list)

#     return text_with_pytesseract


# 3. Extract structured info from text via LLM
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)


def load_conversation(filename):

    with open(filename, 'r') as f:
        conversation = f.read()

    return conversation

class HuggingFaceLLM:
    def __init__(self, temperature=0, top_k=2, model_name="bkai-foundation-models/vietnamese-llama2-7b-120GB", cache_dir = '/home/vinbig/Documents/PA_Modeling/Prompt/bk_tmp'):
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=True, device_map="auto")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # torch_dtype=torch.float16,
            # load_in_8bit=True,
            quantization_config=nf4_config,
            # device_map="auto",
            cache_dir=cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.top_k = top_k

    def generate(self, prompt, max_length=1024):
        json = {
            "type": "object",
            "properties": {
                "current_institute": {"type": "string"},
                "name": {"type": "string"},
                "gender": {"type": "string"},
                "birth": {"type": "string"},
                "age": {"type": "string"},
                "address": {"type": "string"},
                "id_bhyt": {"type": "string"},
                "diagnosis": {"type": "string"},
                "date_in": {"type": "string"},
                "doctor_name": {"type": "string"},
            }
        }

        builder = Jsonformer(
            model=self.model,
            tokenizer=self.tokenizer,
            json_schema=json,
            prompt= prompt,
            max_string_token_length=512
        )

        

        print("Generating...")
        output = builder()
        highlight_values(output)
        print(output)
        print("ok")
        return output
    
def extract_structured_data(content: str, data_points):
    llm = HuggingFaceLLM(temperature=0)  # Choose the desired Hugging Face model
    
    template = """
    Bạn là chuyên gia trong lĩnh vực y tế. Mục tiêu của bạn là cung cấp cho người dùng thông tin được trích xuất từ đơn thuốc. Hãy suy nghĩ từng bước một và đừng bao giờ bỏ qua bước nào.

    {content}


    Trên đây là nội dung; vui lòng thử trích xuất tất cả các điểm dữ liệu từ nội dung trên:
    {data_points}
    """

    # Fill in the placeholders in the template
    formatted_template = template.format(content=content, data_points=data_points)
    #print(formatted_template)
    
    # Generate text using the formatted template
    
    results = llm.generate(formatted_template)

    return results

def main():
    default_data_points = """{
        "current_institute": "Tên bệnh viện/ Tên phòng khám phát hành đơn thuốc",
        "name": "Tên bệnh nhân",
        "gender": "Giới tính của bệnh nhân",
        "birth": "(ngày, tháng) năm sinh bệnh nhân",
        "age": "Tuổi bệnh nhân",
        "address": "Địa chỉ bệnh nhân",
        "tel_customer": "Số điện thoại của bệnh nhân",
        "id_bhyt": "Số thẻ bảo hiểm y tế của bệnh nhân",
        "diagnosis": "Chẩn đoán bệnh",
        "date_in": "Ngày phát hành đơn thuốc",
        "doctor_name": "Họ tên Bác sĩ kê đơn",
    }"""

    # st.set_page_config(page_title="Doc extraction", page_icon=":bird:")

    # st.header("Doc extraction :bird:")

    # data_points = st.text_area(
    #     "Data points", value=default_data_points, height=170)

    # folder_path = './pdfs'  # Replace this with your folder path containing PDFs

    # pdf_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith('.pdf')]

    results = []
    # if pdf_paths:
    #     total_start_time = time.time()
    #     with open("output_results.txt", "w") as output_file:
    #         for pdf_path in pdf_paths:
    #             with NamedTemporaryFile(dir='.', suffix='.csv') as f:
    #                 output_file.write(f"PDF Path: {pdf_path}\n")
    #                 start_time = time.time()  # Record the start time
    #                 content = extract_content_from_url(pdf_path)
    #                 data = extract_structured_data(content, default_data_points)
    #                 json_data = json.dumps(data)
    #                 if isinstance(json_data, list):
    #                     results.extend(json_data)
    #                 else:
    #                     results.append(json_data)
    #                 end_time = time.time()  # Record the end time
    #                 elapsed_time = end_time - start_time
    #                 output_file.write(f"Execution time: {elapsed_time:.2f} seconds\n")
    #                 output_file.write(f"Results: {json_data}\n")
    #                 output_file.write("\n")
    #         total_end_time = time.time()
    #         total_elapsed_time = total_end_time - total_start_time
    #         output_file.write(f"Total execution time: {total_elapsed_time:.2f} seconds\n")
    path_input = '/home/vinbig/Documents/PA_Modeling/Prompt/private_test_Pharma/ocr_text/Long_Chau_414.txt'
    input = load_conversation(path_input)
    data = extract_structured_data(input, default_data_points)
    json_data = json.dumps(data, ensure_ascii=False)
    if isinstance(json_data, list):
        results.extend(json_data)
    else:
        results.append(json_data)
    # end_time = time.time()  # Record the end time
    # elapsed_time = end_time - start_time
    # output_file.write(f"Execution time: {elapsed_time:.2f} seconds\n")
    # output_file.write(f"Results: {json_data}\n")
    # output_file.write("\n")
        

if __name__ == '__main__':
    # multiprocessing.freeze_support()
    main()
