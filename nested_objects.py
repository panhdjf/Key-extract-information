from typing import List, Optional
import pandas as pd

from langchain.schema import Document
from langchain.callbacks import get_openai_callback

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import HuggingFacePipeline

import streamlit as st
import torch
from transformers import  AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer, pipeline
from kor.nodes import Object, Text, Number
from kor.extraction import create_extraction_chain
from kor import extract_from_documents, from_pydantic, create_extraction_chain
from transformers import AutoConfig, BitsAndBytesConfig




import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
import asyncio
device = "cuda:0" if torch.cuda.is_available() else "cpu"

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# cache_dir = '/home/vinbig/Documents/PA_Modeling/Prompt/tmp'
# tokenizer = AutoTokenizer.from_pretrained("vilm/vinallama-7b", cache_dir=cache_dir)
# model = AutoModelForCausalLM.from_pretrained("vilm/vinallama-7b",cache_dir=cache_dir, quantization_config=nf4_config)

cache_dir = '/home/vinbig/Documents/PA_Modeling/Prompt/bk_tmp'
tokenizer = AutoTokenizer.from_pretrained("bkai-foundation-models/vietnamese-llama2-7b-120GB", cache_dir=cache_dir)
# model = AutoModelForCausalLM.from_pretrained("bkai-foundation-models/vietnamese-llama2-7b-120GB", cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained("bkai-foundation-models/vietnamese-llama2-7b-120GB",cache_dir=cache_dir, quantization_config=nf4_config)



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

def load_conversation(filename):

    with open(filename, 'r') as f:
        conversation = f.read()

    return conversation

conversation = load_conversation('/home/vinbig/Documents/PA_Modeling/Prompt/longchau_526.txt')
doc = Document(page_content=conversation)
split_docs = RecursiveCharacterTextSplitter().split_documents([doc])


patient = Object(
    id="patient",
    description = 'Thông tin về bệnh nhân, người được kê đơn thuốc',
    attributes=[
        Text(id='current_institute', description = 'Tên bệnh viện/ Tên phòng khám'),
        Text(id='name', description = 'Họ và tên của bệnh nhân'),
        Text(id='gender', description = 'Giới tính của bệnh nhân'),
        Text(id='birth', description = '(ngày, tháng) năm sinh bệnh nhân'),
        Text(id='age', description = 'Tuổi bệnh nhân'),
        Text(id='address', description = 'Địa chỉ/ Địa chỉ thường chú/ Địa chỉ tạm chú của bệnh nhân'),
        Text(id='tel_customer', description = 'Số điện thoại của bệnh nhân' ),
        Text(id='id_bhyt', description = 'Số thẻ bảo hiểm y tế của bệnh nhân'),
        # Text(id='CCCD', description = 'Số CCCD/CMND'),
        # Text(id='allergy', description = 'Dị ứng (đồ ăn/ thành phần thuốc)'),
        Text(id='diagnosis', description = 'Chẩn đoán bệnh'),
        Text(id='date_in', description = 'Ngày phát hành đơn thuốc'),
        Text(id='doctor_name', description = 'Họ tên Bác sĩ kê đơn'),
    ],
    examples=[
        (
            "ĐƠN THUỐC 01/BV-01\nBệnh viện Hữu Nghị Lạc Việt\n\nTên bệnh nhân: Nguyễn Văn An 45 tuổi \nGiới tính Nam   Số ĐT \n0374685383 GIẤY RA VIỆN\nGhi chú: uống thuốc theo đơn.Tái khám sạu 01 tháng hoặc khi có bất thường Ngày 03 tháng 05 năm 2023 ",
            {"current_institute": "Bệnh viện Hữu Nghị Lạc Việt", "name": "Nguyễn Văn An", "gender":"Nam" , "birth" : "", "age": "45", "address" : "" , "tel_customer" : "0374685383", "id_bhyt" : "", "diagnosis" : "", "date_in" : "Ngày 03 tháng 05 năm 2023", "doctor_name" : ""}

        ),
        (
            "Sở y tế Vĩnh Phúc \n BỆNH BIỆN ĐA KHOA TỈNH \n\n ĐƠN THUỐC BẢO HIỂM Họ và tên Trần Thị Thoán \n     Tuổi \n25 Địa chỉ    Xã Trần Hưng Đạo - Huyện Lý Nhân - Tỉnh Hà Nam \nChẩn đoán\n Đau dạ dày Số thẻ BHYT:\nBA3763373626123 Lưu ý: Khám lại khi thấy bắt thường và khi hết thuốc.  Ngày 18 tháng 10 năm 2021 Kế toán\nThủ kho\n(Ký và ghi rõ họ, tên)\nNgười bệnh\n(Ký và ghi rõ họ, tên) Bác sĩ khám\nCor.\nLê Văn Chinh",
            {"current_institute": "Sở y tế Vĩnh Phúc BỆNH BIỆN ĐA KHOA TỈNH ", "name": "Trần Thị Thoán", "gender":"", "birth" : "", "age": "25", "address": "Xã Trần Hưng Đạo - Huyện Lý Nhân - Tỉnh Hà Nam", "tel_customer" : "", "id_bhyt" : "BA37633567673626", "diagnosis" : "Đau dạ dày, Khó thở, viêm phổi", "date_in":"Ngày 18 tháng 10 năm 2021", "doctor_name": "Lê Văn Chinh"}
        )
    ],
)

# general_prescription = Object(
#     id = "general_prescription",
#     description = 'Thông tin chung về Đơn thuốc',
#     attributes=[
#         Text(id='current_institute', description = 'Tên bệnh viện/ Tên phòng khám'),
#         Text(id='date_in', description = 'Ngày ký giấy phát đơn thuốc'),
#         Text(id='date_out', description = 'Ngày đơn thuốc hết hiệu lực'),
#         Text(id='doctor_name', description = 'Họ tên bác sĩ kê và ký đơn'),
#         Text(id='doctor_sign', description = 'Bác sỹ có ký xác thực không (Có hoặc Không)'),
#     ],
#     examples=[
#         (
#             "01/BV-01\nBệnh viện Hữu Nghị Lạc Việt\n\nGIẤY RA VIỆN\nGhi chú: uống thuốc theo đơn.Tái khám sạu 01 tháng hoặc khi có bất thường Ngày 03 tháng 05 năm 2023",
#             {
#                 "current_institute": "Bệnh viện Hữu Nghị Lạc Việt", 
#                 "date_in":"Ngày 03 tháng 05 năm 2023",
#                 "date_out" : "" , 
#                 "doctor_name": "", 
#                 "doctor_sign" : ""
#             },
#         ),
#         (
#             "Lưu ý: Khám lại khi thấy bắt thường và khi hết thuốc.  Ngày 18 tháng 10 năm 2021 Kế toán\nThủ kho\n(Ký và ghi rõ họ, tên)\nNgười bệnh\n(Ký và ghi rõ họ, tên) Bác sĩ khám\nCor.\nLê Văn Chinh",
#             {
#                 "current_institute": "", 
#                 "date_in":"Ngày 18 tháng 10 năm 2021",
#                 "date_out" : "" , 
#                 "doctor_name": "Lê Văn Chinh", 
#                 "doctor_sign" : "True"
#             },
#         )
#     ],
# )
# schema = Object(
#     id = 'information',
#     description="Bạn là một dược sĩ trung thực. Từ đơn thuốc sau, hãy trích xuất thông tin về được yêu cầu bên dưới đây. Không được thêm hay bớt bất kỳ không tin nào. Nếu không biết, bạn chỉ cần trả lời không biết, không được đưa thông tin không có trong tài liệu vào câu trả lời.",
#     attibutes=[
#         Text(
#             id="drug_name",
#             description="Tên thuốc, thành phần hoạt chất, hàm lượng",
#             examples=[("Paracetamol 500mg sáng tối 1 viên", "Paracetamol 500mg")],
#         ),
#         patient,
#         general_prescription,
#     ],
#     many=True,
# )

# chain = create_extraction_chain(
#     llm, schema, encoder_or_encoder_class="json", input_formatter=None
# )

schema = Object(
    id="information",
    description="Bạn là một dược sĩ trung thực. Tài liệu sau đây là thông tin của một đơn thuốc, hãy trích xuất thông tin của 11 trường bao gômg: {current_institute, name, gender, birth, age, address, tel_customer, id_bhyt, diagnosis, date_in, doctor_name}. Không được thêm hay bớt bất kỳ không tin nào. Nếu không biết, bạn chỉ cần trả lời không biết. Chắc chắn không được đưa thông tin không có trong tài liệu vào câu trả lời.",
    attributes=[
        # Text(
        #     id="person_name",
        #     description="The full name of the person or partial name",
        #     examples=[("John Smith was here", "John Smith")],
        # ),
        patient
    ],
    many=True,
)

chain = create_extraction_chain(
    llm, schema, encoder_or_encoder_class="json", input_formatter=None
)
# print(chain.prompt.format_prompt(text="[user input]").to_string())

async def result():
    # with get_openai_callback() as cb:
    document_extraction_results = await extract_from_documents(
        chain, split_docs, max_concurrency=5, use_uid=False, return_exceptions=True,
    )
    return document_extraction_results
document_extraction_results = asyncio.run(result())
print(document_extraction_results)