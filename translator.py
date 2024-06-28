import cv2
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from string import ascii_letters
import textwrap
import os
from PIL import Image
from pdf2image import convert_from_path
from transformers import pipeline

from bubble_detector import detect_bubbles
from utils import find_textbox, get_font_size
#from translator import translate

class ocr_pipeline:
    def __init__(self,ocr_pipename):
        self.ocr_pipename = ocr_pipename
        if ocr_pipename == 'easyocr':
            import easyocr
            os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
            self.ocr_pipeline = easyocr.Reader(['en'])
        elif ocr_pipename == "models/kha-white/manga-ocr-base":
            self.ocr_pipeline = pipeline("image-to-text", model="models/kha-white/manga-ocr-base")
    def answer(self,image):
        if self.ocr_pipename == 'easyocr':
            result = self.ocr_pipeline.readtext(image)
            res = ""
            for (bbox, text, prob) in result:
                res = res + " "+text
            return res
        elif self.ocr_pipename == "models/kha-white/manga-ocr-base":
            image = Image.fromarray(image)
            return self.ocr_pipeline(image)[0]['generated_text']
        
class translation_pipeline:
    def __init__(self,translation_pipename):
        self.translation_pipename = translation_pipename
        if translation_pipename == "models/Helsinki-NLP/opus-mt-ja-en":
            self.translation_pipeline = pipeline("translation", model=translation_pipename)
        elif translation_pipename == "models/Helsinki-NLP/opus-mt-en-ml":
            self.translation_pipeline = pipeline("translation", model=translation_pipename)
    def answer(self,text):
        if self.translation_pipename == "models/Helsinki-NLP/opus-mt-ja-en":
            return self.translation_pipeline(text)[0]['translation_text']
        elif self.translation_pipename == "models/Helsinki-NLP/opus-mt-en-ml":
            return self.translation_pipeline(text)[0]['translation_text']
            
class translator:
    def __init__(self, languages, input_file, output_file, ocr_pipename, translation_pipename):
        self.languages = languages
        self.input_file = input_file
        self.output_file = output_file
        self.ocr_pipe = ocr_pipeline(ocr_pipename)
        self.translation_pipe = translation_pipeline(translation_pipename)
        self.font_path = "assets/manga.ttf"       
        
    def translate_image(self,image):
        boxes = detect_bubbles(image)
        balloons = 1
        for x,y,w,h in boxes.xywh.numpy():
            xmin, xmax = int(x - w/2), int(x + w/2)
            ymin, ymax = int(y - h/2), int(y + h/2)
            cropped_image = image[ymin:ymax,xmin:xmax]
            cropped_pil_image = Image.fromarray(cropped_image)
            generated_text = self.ocr_pipe.answer(cropped_image)
            translated_text = self.translation_pipe.answer(generated_text)
            window_x,window_y,window_w,window_h = find_textbox(cropped_image)
            wrapped,font_size = get_font_size((window_w,window_h),translated_text,self.font_path)
            draw = ImageDraw.Draw(cropped_pil_image)
            draw.rectangle((window_x,window_y,window_x+window_w,window_y+window_h), fill = "white")
            font = ImageFont.truetype(self.font_path, font_size)
            x_center = int(window_w / 2) + window_x
            y_center = int(window_h/ 2) + window_y
            draw.text((x_center, y_center), wrapped, font=font, fill="black", anchor="mm",align='center')
            image[ymin:ymax,xmin:xmax] = cv2.cvtColor(np.array(cropped_pil_image), cv2.COLOR_RGB2BGR)
            balloons += 1
            
        return image
    def translate(self):
        input_name, input_ext = os.path.splitext(self.input_file)
        if input_ext.lower() == '.pdf':
            images = convert_from_path(self.input_file)
            im1 = []
            for i, image in enumerate(images):
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                res_image = self.translate_image(image)
                res_image = Image.fromarray(res_image)
                pdf_bytes = res_image.convert('RGB')
                im1.append(pdf_bytes)
            im1[0].save(f'{input_name}_convered.pdf', save_all=True, append_images=im1[1:])
        elif input_ext.lower() in ['.png','.jpg','.jpeg']:
            # Convert Image and rotate
            image = cv2.imread(self.input_file)
            res_image = self.translate_image(image)
            cv2.imwrite(self.output_file,res_image)

            print(f'Image translated successfully. Saved as {output_file}')
        else:
            print("File should be png, jpeg, or pdf")
        
    


