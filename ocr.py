from utils import *
import os
import time

def get_text(ocr_res):
    text = ''
    phrases = ocr_res["phrases"]
    for phrase in phrases:
        text += phrase['text'] + '\n'
    return text

def read_imgs(path_file):
    images = [Image.open(path_file)]

    # get pdf text
    raw_text = ''
    for img in images:
        ocr_res = get_OCR(img, preprocess=False)
        text = get_text(ocr_res)
        raw_text += text
    return raw_text

path_imgs = '/home/vinbig/Documents/PA_Modeling/Prompt/prescription_label_text/images'
path_txts = '/home/vinbig/Documents/PA_Modeling/Prompt/prescription_label_text/ocr_text'
list_names = os.listdir(path_imgs)
# for name in list_names:
#     path_file = os.path.join(path_imgs, name)
#     raw_text = read_imgs(path_file)

#     n = name.split(".")[0] + '.txt'
#     try:
#         f = open(os.path.join(path_txts, n), 'w')
#         f.write(raw_text)
#         f.close()
#     except:
#         print("Cant not write to file txt")
    
#     time.sleep(1)

path_img = "/home/vinbig/Documents/PA_Modeling/Prompt/prescription_label_text/images/Long_Chau_28.jpg"
path_save = "/home/vinbig/Documents/PA_Modeling/Prompt/Long_Chau_28.txt"
raw_text = read_imgs(path_img)
f = open(path_save, 'w')
f.write(raw_text)