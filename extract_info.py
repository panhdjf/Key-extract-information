from typing import List, Optional
from langchain import HuggingFacePipeline
from kor.nodes import Object, Text, Number
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, OutputFixingParser
from langchain.chains.openai_functions.extraction import create_extraction_chain_pydantic
from langchain.prompts.few_shot import FewShotPromptTemplate


import pandas as pd
from pydantic import BaseModel, Field, validator
from kor import extract_from_documents, from_pydantic, create_extraction_chain

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from transformers import  AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import TextStreamer, pipeline
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
import asyncio
from transformers import AutoConfig, BitsAndBytesConfig
from langchain.output_parsers import PydanticOutputParser

from pdf2image import convert_from_bytes
from PIL import Image
from utils import get_OCR

device = "cuda:0" if torch.cuda.is_available() else "cpu"

'''----LOAD MODEL----'''
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# '''----MODEL VietnamAIHub/Vietnamese_llama2------'''
# model_name_or_path = "VietnamAIHub/Vietnamese_llama2_7B_8K_SFT_General_domain"
# cache_dir = '/home/vinbig/Documents/PA_Modeling/Retrieval_pdf/tmp1'
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name_or_path,
#     torch_dtype=torch.float32,
#     pretraining_tp=1,
#     cache_dir=cache_dir,
# )


# '''----MODEL vilm/vinallama-7b---------'''
# cache_dir = '/home/vinbig/Documents/PA_Modeling/Prompt/tmp_vinallama'
# tokenizer = AutoTokenizer.from_pretrained("vilm/vinallama-7b", cache_dir=cache_dir)
# # model = AutoModelForCausalLM.from_pretrained("vilm/vinallama-7b",cache_dir=cache_dir, quantization_config=nf4_config)
# model = AutoModelForCausalLM.from_pretrained("vilm/vinallama-7b",cache_dir=cache_dir)

'''----MODEL bkai-foundation-models/vietnamese-llama2---------'''
cache_dir = '/home/vinbig/Documents/PA_Modeling/Prompt/bk_tmp'
tokenizer = AutoTokenizer.from_pretrained("bkai-foundation-models/vietnamese-llama2-7b-120GB", cache_dir=cache_dir)
# model = AutoModelForCausalLM.from_pretrained("bkai-foundation-models/vietnamese-llama2-7b-120GB", cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained("bkai-foundation-models/vietnamese-llama2-7b-120GB",cache_dir=cache_dir, quantization_config=nf4_config)


# '''----MODEL PhoGPT instruct---------'''

# cache_dir = '/home/vinbig/Documents/PA_Modeling/Prompt/pho_gpt'
# model_path = "vinai/PhoGPT-7B5-Instruct"  
  
# config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)  
# config.init_device = "cuda"
# # config.attn_config['attn_impl'] = 'triton' # Enable if "triton" installed!
  
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     config=config,
#     trust_remote_code=True,
#     quantization_config=nf4_config,
#     cache_dir=cache_dir
# )
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_dir)  

# '''----MODEL PhoGPT ---------'''
# cache_dir = '/home/vinbig/Documents/PA_Modeling/Prompt/phogpt_tmp'
# model = AutoModel.from_pretrained("vinai/PhoGPT-7B5", quantization_config=nf4_config, cache_dir=cache_dir)
# tokenizer = AutoTokenizer.from_pretrained("vinai/PhoGPT-7B5", trust_remote_code=True, cache_dir=cache_dir)  

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.01,
    top_p=0.95,
    repetition_penalty=1.15,
    streamer=streamer,
)

llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.01, "max_length":512, "stop": "</exp>"})


'''-----------LOAD DOCUMENT---------------'''
def load_conversation(filename):

    with open(filename, 'r') as f:
        conversation = f.read()

    return conversation

# conversation = load_conversation('/home/vinbig/Documents/PA_Modeling/Prompt/bv_th_1009.txt')
def get_text_chunks(text):
    doc = Document(page_content=text)
    split_docs = RecursiveCharacterTextSplitter().split_documents([doc])
    return split_docs

'''--------------PROMPT------------'''
class Patient(BaseModel):
    current_institute: Optional[str] = Field(
        description="Tên bệnh viện/ Tên phòng khám phát hành đơn thuốc",
    )
    name: str = Field(
        description="Tên bệnh nhân",
    )
    gender: Optional[str] = Field(
        description="Giới tính của bệnh nhân",
    )
    birth: Optional[str] = Field(
        description="(ngày, tháng) năm sinh bệnh nhân",
    )
    age: Optional[str] = Field(
        description="Tuổi bệnh nhân",
    )

    address: Optional[str] = Field(
        description="Địa chỉ bệnh nhân",
    )
    tel_customer: Optional[str] = Field(
        description="Số điện thoại của bệnh nhân",
    )
    id_bhyt: Optional[str] = Field(
        description="Số thẻ bảo hiểm y tế của bệnh nhân",
    )
    # CCCD: Optional[str] = Field(
    #     description="Số CCCD/CMND của bệnh nhân",
    # )
    # allergy: Optional[str] = Field(
    #     description="Dị ứng của bệnh nhân",
    # )
    diagnosis: Optional[str] = Field(
        description="Chẩn đoán bệnh",
    )

    date_in: Optional[str] = Field(
        description="Ngày phát hành đơn thuốc",
    )
    # date_out: Optional[str] = Field(
    #     description="ngày đơn thuốc hết hiệu lực",
    # )
    doctor_name: Optional[str] = Field(
        description="Họ tên Bác sĩ kê đơn",
    )

    # @validator("name")
    # def name_must_not_be_empty(cls, v):
    #     if not v:
    #         raise ValueError("Tên không được để trống")
    #     return v



schema, extraction_validator = from_pydantic(
    Patient,
    description="Bạn là một dược sĩ trung thực. Hãy trích xuất thông tin của đúng 11 trường sau đây chỉ dựa vào ngữ cảnh là nội dung của một đơn thuốc đưa vào: {current_institute, name, gender, birth, age, address, tel_customer, id_bhyt, diagnosis, date_in, doctor_name}. Phản hồi phải được bắt đầu bằng <exp> và kết thúc bằng </exp>. Không được thêm hay bỏ qua bất kỳ thông tin nào. Nếu không biết, bạn chỉ cần trả lời không biết, không được đưa thông tin không có trong tài liệu vào câu trả lời.",
    examples=[
        (
            "ĐƠN THUỐC 01/BV-01\nBệnh viện Hữu Nghị Lạc Việt\n\nTên bệnh nhân: Nguyễn Văn An 45 tuổi \nGiới tính Nam   Số ĐT \n0374685383 GIẤY RA VIỆN\nGhi chú: uống thuốc theo đơn.Tái khám sạu 01 tháng hoặc khi có bất thường Ngày 03 tháng 05 năm 2023 ",
            # {"current_institute": "Bệnh viện Hữu Nghị Lạc Việt", "name": "Nguyễn Văn An", "gender":"Nam" , "birth" : "", "age": "45", "address" : "" , "tel_customer" : "0374685383", "id_bhyt" : "", "diagnosis" : "", "date_in" : "Ngày 03 tháng 05 năm 2023", "doctor_name" : ""}
            """<exp>{"current_institute": "Bệnh viện Hữu Nghị Lạc Việt", "name": "Nguyễn Văn An", "gender":"Nam" , "birth" : "", "age": "45", "address" : "" , "tel_customer" : "0374685383", "id_bhyt" : "", "diagnosis" : "", "date_in" : "Ngày 03 tháng 05 năm 2023", "doctor_name" : ""}</exp>"""

        ),
        # (
        #     " BỆNH BIỆN ĐA KHOA TỈNH \n\n ĐƠN THUỐC BẢO HIỂM Họ và tên Trần Thị Thoán \n   Năm sinh: 1998   Tuổi \n25 Địa chỉ    Xã Trần Hưng Đạo - Huyện Lý Nhân - Tỉnh Hà Nam \nChẩn đoán\n Đau dạ dày, Khó thở, viêm phổi Số thẻ phòng khám 10A1 BHYT:\nBA3763373626123 Lưu ý: Khám lại khi thấy bắt thường và khi hết thuốc.  Ngày 18 tháng 10 năm 2021 Kế toán\nThủ kho\n(Ký và ghi rõ họ, tên)\nNgười bệnh\n(Ký và ghi rõ họ, tên) Bác sĩ khám\nCor.\nLê Văn Chinh",
        #     {"current_institute": "BỆNH BIỆN ĐA KHOA TỈNH ", "name": "Trần Thị Thoán", "gender":"", "birth" : "1998", "age": "25", "address": "Xã Trần Hưng Đạo - Huyện Lý Nhân - Tỉnh Hà Nam", "tel_customer" : "", "id_bhyt" : "nBA3763373626123", "diagnosis" : "Đau dạ dày, Khó thở, viêm phổi", "date_in" : "Ngày 18 tháng 10 năm 2021", "doctor_name" : "Lê Văn Chinh"}
        # ),
        # (
        #     "\n Đường Lê Văn Lương \n Bệnh Viện Mắt Trung Ương  Họ và tên: Phạm Văn Bê \n Tuổi: 50 Phái: Nam  Năm sinh: 1973 \nBảo Hiểm Y Tế \n Số ĐT: 03462648261 \n\nSố thẻ BHYT: MK338745874850166 Địa chỉ \n SN76 Hai Bà Trưng, TP Hà Nội \n\n Chẩn đooán: Bệnh trào ngược dạ dày- thực quản / Cơn đau thắt ngực ổn định ",
        #     {"current_institute": "Bệnh Viện Mắt Trung Ương", "name": "Phạm Văn Bê", "gender":"Nam", "birth" : "1973", "age": "50", "address": "SN76 Hai Bà Trưng, TP Hà Nội", "tel_customer" : "03462648261", "id_bhyt" : "MK338745874850166", "diagnosis" : "Bệnh trào ngược dạ dày- thực quản / Cơn đau thắt ngực ổn định", "date_in":"", "doctor_name": ""}
        # )
    ],
    many=True,
    )


# parser = PydanticOutputParser(pydantic_object=Patient)
# format_instructions = parser.get_format_instructions()

# examples = [
#     {
#         'question': "ĐƠN THUỐC 01/BV-01\nBệnh viện Hữu Nghị Lạc Việt\n\nTên bệnh nhân: Nguyễn Văn An 45 tuổi \nGiới tính Nam   Số ĐT \n0374685383 GIẤY RA VIỆN\nGhi chú: uống thuốc theo đơn.Tái khám sạu 01 tháng hoặc khi có bất thường Ngày 03 tháng 05 năm 2023 BS. ", 
#         'answer': 
# '''
# <exp>"current_institute": "Bệnh viện Hữu Nghị Lạc Việt", "name": "Nguyễn Văn An", "gender":"Nam" , "birth" : "", "age": "45", "address" : "" , "tel_customer" : "0374685383", "id_bhyt" : "", "diagnosis" : "", "date_in" : "Ngày 03 tháng 05 năm 2023", "doctor_name" : ""</exp>
# '''
#     },
#     {
#         'question': " BỆNH BIỆN ĐA KHOA TỈNH \n\n ĐƠN THUỐC BẢO HIỂM Họ và tên Trần Thị Thoán \n   Năm sinh: 1998   Tuổi \n25 Địa chỉ    Xã Trần Hưng Đạo - Huyện Lý Nhân - Tỉnh Hà Nam \nChẩn đoán\n Đau dạ dày, Khó thở, viêm phổi Số thẻ phòng khám 10A1 BHYT:\nBA3763373626123 Lưu ý: Khám lại khi thấy bắt thường và khi hết thuốc.  Ngày 18 tháng 10 năm 2021 Kế toán\nThủ kho\n(Ký và ghi rõ họ, tên)\nNgười bệnh\n(Ký và ghi rõ họ, tên) Bác sĩ khám\nCor.\nLê Văn Chinh",
#         'answer':
# '''
# <exp>"current_institute": "BỆNH BIỆN ĐA KHOA TỈNH ", "name": "Trần Thị Thoán", "gender":"", "birth" : "1998", "age": "25", "address": "Xã Trần Hưng Đạo - Huyện Lý Nhân - Tỉnh Hà Nam", "tel_customer" : "", "id_bhyt" : "nBA3763373626123", "diagnosis" : "Đau dạ dày, Khó thở, viêm phổi", "date_in" : "Ngày 18 tháng 10 năm 2021", "doctor_name" : "Lê Văn Chinh"</exp>
# '''
#     },
#     {
#         'question': "\n Đường Lê Văn Lương \n Bệnh Viện Mắt Trung Ương  Họ và tên: Phạm Văn Bê \n Tuổi: 50 Phái: Nam  Năm sinh: 1973 \nBảo Hiểm Y Tế \n Số ĐT: 03462648261 \n\nSố thẻ BHYT: MK338745874850166 Địa chỉ \n SN76 Hai Bà Trưng, TP Hà Nội \n\n Chẩn đooán: Bệnh trào ngược dạ dày- thực quản / Cơn đau thắt ngực ổn định ",
#         'answer':
# '''
# <exp>"current_institute": "Bệnh Viện Mắt Trung Ương", "name": "Phạm Văn Bê", "gender":"Nam", "birth" : "1973", "age": "50", "address": "SN76 Hai Bà Trưng, TP Hà Nội", "tel_customer" : "03462648261", "id_bhyt" : "MK338745874850166", "diagnosis" : "Bệnh trào ngược dạ dày- thực quản / Cơn đau thắt ngực ổn định", "date_in":"", "doctor_name": ""</exp>
# '''
#     }
# ]


# example_prompt = PromptTemplate(input_variables=["question", "answer"], template="Question: {question} Phản hồi phải được bắt đầu bằng {<exp>} và kết thúc bằng </exp> \n{answer}")

# # example_prompt = PromptTemplate(
# #     template="Trích xuất thông tin theo mẫu..\n{format_instructions}\nQuestion: {question}\n{answer}\n",
# #     input_variables=["question", "answer"],
# #     partial_variables={"format_instructions": format_instructions},
# # )

# prompt = FewShotPromptTemplate(
#     examples=examples,
#     example_prompt=example_prompt,
#     suffix="Question: {inputText}",
#     input_variables=["inputText"], 
# )

# # prompt = PromptTemplate(
# #     template="Trích xuất tin từ một hóa đơn thuốc.\n{format_instructions}\nCâu trả lời phải được trình bày dưới dạng khối mã JSON đánh dấu.\nHóa đơn thuốc: {inputText}\n",
# #     input_variables=["inputText"],
# #     partial_variables={"format_instructions": format_instructions},
# # )

# text = load_conversation('/home/vinbig/Documents/PA_Modeling/Prompt/private_test_Pharma/ocr_text/Long_Chau_46.txt')
# split_docs = get_text_chunks(text)

# input_text = prompt.format(inputText = split_docs[0].page_content)


# chain = create_extraction_chain_pydantic(Patient, llm, verbose=True)
# print('--')
# chain.run(input_text)
# print('--')

# output = parser.parse(product_details)
# print(product_details)




chain = create_extraction_chain(
    llm,
    schema,
    # encoder_or_encoder_class="csv",
    validator=extraction_validator,
    # encoder_or_encoder_class="json", 
    input_formatter=None, 
)

# path = '/home/vinbig/Documents/PA_Modeling/Prompt/private_test_Pharma/ocr_text'
# name_files = os.listdir(path)

# text =  load_conversation(os.path.join(path, name_files[1]))

text = load_conversation('/home/vinbig/Documents/PA_Modeling/Prompt/private_test_Pharma/ocr_text/Long_Chau_46.txt')
split_docs = get_text_chunks(text)
# print('\n---------------')
# print(os.path.join(path, name_files[201]))
# print(split_docs)
# print('---------------\n')


async def result():
    # with get_openai_callback() as cb:
    document_extraction_results = await extract_from_documents(
        chain, split_docs, max_concurrency=5, use_uid=False, return_exceptions=True,
    )
    return document_extraction_results
document_extraction_results = asyncio.run(result())
# print("-----------------")
# raw_data = document_extraction_results
# print(raw_data[0]['raw'])


