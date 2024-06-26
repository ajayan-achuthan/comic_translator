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
class veruthe:
    def __init__(self):
        self.nice =0
class translator:
    def __init__(self, languages, input_file, output_file, ocr_pipeline, translation_pipeline):
        self.languages = languages
        self.input_file = input_file
        self.output_file = output_file
        self.ocr_pipe = pipeline("image-to-text", model=ocr_pipeline)
        self.translation_pipe = pipeline("translation", model=translation_pipeline)
        self.font_path = "assets/manga.ttf"       
        
    def translate_image(self,image):
        boxes = detect_bubbles(image)
        balloons = 1
        for x,y,w,h in boxes.xywh.numpy():
            xmin, xmax = int(x - w/2), int(x + w/2)
            ymin, ymax = int(y - h/2), int(y + h/2)
            cropped_image = image[ymin:ymax,xmin:xmax]
            cropped_pil_image = Image.fromarray(cropped_image)
            generated_text = self.ocr_pipe(cropped_pil_image)[0]['generated_text']
            translated_text = self.translation_pipe(generated_text)[0]['translation_text']
            print(generated_text,translated_text)
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

            print('Image translated successfully. Saved as {output_file}')
        else:
            print("File should be png, jpeg, or pdf")
        
    


