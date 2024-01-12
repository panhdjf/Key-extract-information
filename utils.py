import gc

from PIL import Image, ImageOps, ImageDraw
from io import BytesIO
import base64
import requests


def encode_image(img_pil):
    buffered = BytesIO()
    # img_pil = Image.fromarray(np.uint8(image_np))
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    img_base64 = img_str.decode('utf-8')
    # img_base64 = f"data:image/png;base64,{img_base64}"

    del buffered 
    del img_pil
    del img_str
    gc.collect()

    return img_base64


def decode_image(image_b64, mode=None):
    # img_encode = image_b64.split(',')[1]
    img_bytes = base64.b64decode(image_b64)
    # img = Image.open(BytesIO(img_bytes))
    if mode is not None:
        img = Image.open(BytesIO(img_bytes)).convert(mode)
    else:
        img = Image.open(BytesIO(img_bytes))
    # img = ImageOps.exif_transpose(img)

    del img_bytes
    gc.collect()

    return img


def get_OCR(img_pil, preprocess=True):
    img_base64 = encode_image(img_pil)
    try:
        r = requests.post('http://10.124.69.99:10000/infer', json={'base64_image': img_base64}, params=dict(preprocess=preprocess))
        r = r.json()
        print('OK------------------------------------------')    
    except Exception as e:
        # print(str(e))
        # print(r.status_code)
        # print(r.status_code)    
        print('Err----------------------------------------------')    
        return []
    
    return r

def convert_json(data):
    new_d = {
        'image_size': data['image_size']
    }
    phrases = data['phrases']
    new_phs = []
    for phrase in phrases:
        words = phrase['words']
        for word in words:
            new_w = [{
                'text': word['text'],
                'bbox': word['bbox']
            }]
            new_phs.append({
                'words': new_w,
                'text': word['text'],
                'bbox':  word['bbox'],
                # 'label': phrase['label'],
                # 'is_key': phrase['is_key'],
                # 'is_value': phrase['is_value']
            })
    new_d['phrases'] = new_phs
    return new_d