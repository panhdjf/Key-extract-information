
from dotenv import load_dotenv
# from pytesseract import image_to_string
# import pypdfium2 as pdfium
import multiprocessing
# from tempfile import NamedTemporaryFile
import json
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
# from tempfile import NamedTemporaryFile
from jsonformer.main import Jsonformer

load_dotenv()

# 1. Read Text
def load_conversation(filename):

    with open(filename, 'r', encoding='utf8') as f:
        conversation = f.read()

    return conversation

# 3. Extract structured info from text via LLM
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

def decode_unicode_escape(data):
    if isinstance(data, dict):
        return {key: decode_unicode_escape(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [decode_unicode_escape(item) for item in data]
    elif isinstance(data, str):
        return eval('"' + data + '"')
    else:
        return data

# model_id = 'LR-AI-Labs/vbd-llama2-7B-50b-chat'
# cache_dir = '/home/vinbig/Documents/PA_Modeling/Prompt/vbd_tmp'
    
# model_id = 'mistralai/Mixtral-8x7B-v0.1'
# cache_dir = '/u01/tmp/mistral_8_7b'
    
# model_id = 'meta-llama/Llama-2-7b-chat-hf'
# cache_dir = '/home/vinbig/Documents/PA_Modeling/Prompt/llama2_tmp'
    
# model_id = 'bkai-foundation-models/vietnamese-llama2-7b-120GB'
# cache_dir = '/home/vinbig/Documents/PA_Modeling/Prompt/bk_tmp'
    
model_id = 'mychen76/mistral7b_ocr_to_json_v1'
cache_dir = '/u01/tmp/mistral_7b_json'

    
class HuggingFaceLLM:
    def __init__(self, temperature=0, top_k=2, model_name=model_id, cache_dir = cache_dir):
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=True, device_map="auto")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # torch_dtype=torch.float16,
            # load_in_8bit=True,
            quantization_config=nf4_config,
            # device_map="auto",
            cache_dir=cache_dir,
            
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.top_k = top_k

    def generate(self, prompt, max_length=512):
        json_schema = {
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
            json_schema=json_schema,
            prompt= prompt,
            max_string_token_length=max_length
        )


        print("Generating...")
        output = builder()
        # output = decode_unicode_escape(output)
        # highlight_values(output)
        print("11", output)
        print("ok")
        return output
    
def extract_structured_data(content: str, data_points):

    # model_id = "bkai-foundation-models/vietnamese-llama2-7b-120GB"
    # cache_dir = '/home/vinbig/Documents/PA_Modeling/Prompt/bk_tmp'

    # model_id = 'mistralai/Mistral-7B-v0.1'
    # cache_dir = '/home/vinbig/Documents/PA_Modeling/Prompt/mistral_tmp'

    # model_id = 'mistralai/Mixtral-8x7B-v0.1'
    # cache_dir = '/home/vinbig/Documents/PA_Modeling/Prompt/mistral_tmp_1'
    llm = HuggingFaceLLM(temperature=0)  # Choose the desired Hugging Face model
    
    template = """
    You are an expert in the medical field. Your goal is to provide the user with information extracted from the prescription. Think step by step and never skip any step.
    
    EXAMPLES
    ----
    Human: ĐƠN THUỐC 01/BV-01\tBệnh viện Hữu Nghị Lạc Việt\t\tTên bệnh nhân: Nguyễn Văn An 45 tuổi \tGiới tính Nam   Số ĐT \t0374685383 GIẤY RA VIỆN\tGhi chú: uống thuốc theo đơn.Tái khám sạu 01 tháng hoặc khi có bất thường Ngày 03 tháng 05 năm 2023 
    AI ASSISTANT: {{"current_institute": "Bệnh viện Hữu Nghị Lạc Việt", "name": "Nguyễn Văn An", "gender":"Nam" , "birth" : "", "age": "45", "address" : "" , "tel_customer" : "0374685383", "id_bhyt" : "", "diagnosis" : "", "date_in" : "Ngày 03 tháng 05 năm 2023", "doctor_name" : ""}}

    Human:  BỆNH BIỆN ĐA KHOA TỈNH \t\t ĐƠN THUỐC BẢO HIỂM Họ và tên Trần Thị Thoán \t   Năm sinh: 1998   Tuổi \t25 Địa chỉ    Xã Trần Hưng Đạo - Huyện Lý Nhân - Tỉnh Hà Nam \tChẩn đoán\t Đau dạ dày, Khó thở, viêm phổi Số thẻ phòng khám 10A1 BHYT:\tBA3763373626123 Lưu ý: Khám lại khi thấy bắt thường và khi hết thuốc.  Ngày 18 tháng 10 năm 2021 Kế toán\tThủ kho\t(Ký và ghi rõ họ, tên)\tNgười bệnh\t(Ký và ghi rõ họ, tên) Bác sĩ.\tLê Văn Chinh
    AI ASSISTANT: {{"current_institute": "BỆNH BIỆN ĐA KHOA TỈNH ", "name": "Trần Thị Thoán", "gender":"", "birth" : "1998", "age": "25", "address": "Xã Trần Hưng Đạo - Huyện Lý Nhân - Tỉnh Hà Nam", "tel_customer" : "", "id_bhyt" : "nBA3763373626123", "diagnosis" : "Đau dạ dày, Khó thở, viêm phổi", "date_in" : "Ngày 18 tháng 10 năm 2021", "doctor_name" : "Lê Văn Chinh"}}

    BEGIN! Answer the Human's question as best as you are able.
    ----

    Here is the content you need to analyze:
    {content}

    Above is the content; please try to extract all data points from the above content. Do not add or omit any information. If you don't know, just answer "don't know" and do not include information that is not in the document in your answer:
    {data_points}
    """


#    Here are some examples:
#         Question: "ĐƠN THUỐC 01/BV-01\nBệnh viện Hữu Nghị Lạc Việt\n\nTên bệnh nhân: Nguyễn Văn An 45 tuổi \nGiới tính Nam   Số ĐT \n0374685383 GIẤY RA VIỆN\nGhi chú: uống thuốc theo đơn.Tái khám sạu 01 tháng hoặc khi có bất thường Ngày 03 tháng 05 năm 2023 "
#         Answer: {{"current_institute": "Bệnh viện Hữu Nghị Lạc Việt", "name": "Nguyễn Văn An", "gender":"Nam" , "birth" : "", "age": "45", "address" : "" , "tel_customer" : "0374685383", "id_bhyt" : "", "diagnosis" : "", "date_in" : "Ngày 03 tháng 05 năm 2023", "doctor_name" : ""}}

        # Question: " BỆNH BIỆN ĐA KHOA TỈNH \n\n ĐƠN THUỐC BẢO HIỂM Họ và tên Trần Thị Thoán \n   Năm sinh: 1998   Tuổi \n25 Địa chỉ    Xã Trần Hưng Đạo - Huyện Lý Nhân - Tỉnh Hà Nam \nChẩn đoán\n Đau dạ dày, Khó thở, viêm phổi Số thẻ phòng khám 10A1 BHYT:\nBA3763373626123 Lưu ý: Khám lại khi thấy bắt thường và khi hết thuốc.  Ngày 18 tháng 10 năm 2021 Kế toán\nThủ kho\n(Ký và ghi rõ họ, tên)\nNgười bệnh\n(Ký và ghi rõ họ, tên) Bác sĩ khám\nCor.\nLê Văn Chinh"
        # Answer: {{"current_institute": "BỆNH BIỆN ĐA KHOA TỈNH ", "name": "Trần Thị Thoán", "gender":"", "birth" : "1998", "age": "25", "address": "Xã Trần Hưng Đạo - Huyện Lý Nhân - Tỉnh Hà Nam", "tel_customer" : "", "id_bhyt" : "nBA3763373626123", "diagnosis" : "Đau dạ dày, Khó thở, viêm phổi", "date_in" : "Ngày 18 tháng 10 năm 2021", "doctor_name" : "Lê Văn Chinh"}}

            
    #         Question: "\n Đường Lê Văn Lương \n Bệnh Viện Mắt Trung Ương  Họ và tên: Phạm Văn Bê \n Tuổi: 50 Phái: Nam  Năm sinh: 1973 \nBảo Hiểm Y Tế \n Số ĐT: 03462648261 \n\nSố thẻ BHYT: MK338745874850166 Địa chỉ \n SN76 Hai Bà Trưng, TP Hà Nội \n\n Chẩn đooán: Bệnh trào ngược dạ dày- thực quản / Cơn đau thắt ngực ổn định "
    #         Answer: {{"current_institute": "Bệnh Viện Mắt Trung Ương", "name": "Phạm Văn Bê", "gender":"Nam", "birth" : "1973", "age": "50", "address": "SN76 Hai Bà Trưng, TP Hà Nội", "tel_customer" : "03462648261", "id_bhyt" : "MK338745874850166", "diagnosis" : "Bệnh trào ngược dạ dày- thực quản / Cơn đau thắt ngực ổn định", "date_in":"", "doctor_name": ""}}

            
    #         Question: 'SUYIE TP. HU CHI MINH\nMS:\n17D/BV-U1\n14:50\nBệnh Viện Nhi Đồng 1\nSố Hồ Sơ:\nPhòng khám:\nB06-TIÊU HÓA\n416336/22\nSố 532 Lý Thái Tổ P10,Q.10\nSố:\n18172256\nPHIÊU TƯ VÂN\nSẢN PHẨM HỒ TRỢ ĐIỀU TRỊ\nNGUYỄN NGỌC THÚY UYÊN\n:35 Tháng\nCân nặng:\n13 Kg CC:cm\nPhái:\nNữ\nchỉ :249 Lê Sao Phường Phú Thạnh, Quận Tân Phú, TP.Hồ Chí Minh\nời thân:ME\nNGUYỄN NGỌC ĐĂNG TUYÊN\nđiện thoại :0933988790\nMacrogol 3350 5g\n10\nGói\n(Bột pha hỗn dich nhuận tràng peGINPOL)\nNgày uống lần, mỗi lần 1 Gói\nInuline+Fructo Oligosaccharide+Gaaact Oligosaccharide 3g\n20 Gói\n(Infogos)\nNgày uống 2 lần, mỗi lần 1 Gói\nkhoản:\n2\n'
    #         Answer: {{'current institute': 'Bệnh viện Nhi Đồng 1', 'name': 'NGUYỄN NGỌC THÚY UYÊN', 'gender': 'Nữ', "birth" : "", "age": "",'address': '249 Lê Sao Phường Phú Thanh, Quận Tân Phú, TP.Hồ Chí Minh', "tel_customer" : "", "id_bhyt" : "", "diagnosis" : "", "date_in":"", "doctor_name": ""}}

            
    #         Question: 'Yersin\nSố 10 Đường Trương Đinh Phường 6, Quận 3 Thành Phố Hồ Chí Min Việt Nam\nPHÒNG KHÁM ĐA KHOA QUỐC TẾ YERSIN\nInternationa\nThe SymbolofSSiile\nWebsite:\nwww.yersinclinic.cmm\nTOA THUỐC\nBệnh nhân:\nPHÙNG THỊ LƯU\nHSBA:\n228061\nNăm sinh:\n1951\nGiới tính:\nNỮ\nĐịa chỉ:\nSỐ 2 HƯNG GIA 1,P. TÂN PHONG, QUẬN7\nChẩn đoán:\nVIÊM DẠ DÀY-HP(+)\n1) NEXIUM MUPS 40mg\nSố lượng:\n60 viên\n(ESOMEPRAZOLE 40mg)\nuống\n02\nlần, mỗi lần\n01 viên - trước ăn 30 phút- 1\ngiờ\n2) AMEBISMO\nSố lượng:\n60 viên\nBISMUTH SUBSALICYLATE 262mg)\này uống\n02\nlần, mỗi lần\n02 viên - nhai nuốt trước ăn 30\nphút- 1 giờ\nTETRACYCLIN 500mg\nSố lượng:\n60 viên\nấy uống\n02\nlần, mỗi lần\n02 viên\nsau ăn 30 phút-1 giờ\nTINIDAZOLE 500mg\nSố lượng:\n30 viên\nuống\n02\nlần, mỗi lần\n01 viên sau ăn 30 phút-1giờ\n) MOTILIUM-M\nSố lượng:\n30 viên\nuống\n02\nlần, mỗi lần\n01 viên - trước ăn 30 phút-1 giờ\nENTEROGERMINA\nSố lượng:\n30 ống\ny uống\n02\nlần, mỗi lần\n01 ống\n- trước ăn 30 phút-1 giò\nKIÊNG ĂN:\nRAU SÓNG, MẮM,\nNgày 28 tháng 11 năm 2023\nNƯỚC CÓ GA, DƯA CÀ MUỐI,\nNGHÊU SÒ ÓC, XOÀI, THƠM,\nBác sĩ điều trị\nSỮA, RƯỢU BIA\nV\nTÁI KHÁM:\n16 /01/ 2023\nBS. TRẦN VĂN HUY\n'
    #         Answer: {{'current institute': 'PHÒNG KHÁM ĐA KHOA QUỐC TẾ YERSIN', 'name': 'PHÙNG THỊ LƯU', 'gender': 'NỮ', 'birth': '1951', "age": "", 'address': '5 ĐƯỜNG SỐ 2, HƯNG GIA 1, P. TÂN PHONG, QUẬN 7', "tel_customer" : "", "id_bhyt" : "", 'diagnosis': 'VIÊM DẠ DÀY - HP (+)', 'date in': 'Ngày 28 tháng 11 năm 2023', 'doctor name': 'BS. TRẦN VĂN HUY'}}

            

    # Fill in the placeholders in the template
    formatted_template = template.format(content=content, data_points=data_points)
    #print(formatted_template)
    
    # Generate text using the formatted template
    
    results = llm.generate(formatted_template)

    return results

def main(path_input):
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
    # results = []
    input = load_conversation(path_input)
    data = extract_structured_data(input, default_data_points)
    json_data = json.dumps(data, ensure_ascii=False)
    # if isinstance(json_data, list):
    #     results.extend(json_data)
    # else:
    #     results.append(json_data)
    # return results    
    return json_data

if __name__ == '__main__':
    multiprocessing.freeze_support()
    path_input = '/home/vinbig/Documents/PA_Modeling/Prompt/private_test_Pharma/ocr_text/Long_Chau_153.txt'

    results = main(path_input)
    print(results)
