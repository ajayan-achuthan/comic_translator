import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from string import ascii_letters
import textwrap

from transformers import pipeline

from bubble_detector import detect_bubbles
from utils import find_textbox, get_font_size
#from translator import translate

if __name__ == "__main__":
    languages = 'jp-en'
    input_image = "input/jp_row.jpg"
    output_image = f'output/{os.path.basename(input_image)}'
    if languages == 'jp-en':
        #japanese
        ocr_pipe = pipeline("image-to-text", model="models/kha-white/manga-ocr-base")
        translation_pipe = pipeline("translation", model="models/Helsinki-NLP/opus-mt-ja-en")
        font_path = "assets/manga.ttf"
    image = cv2.imread(input_image)
    boxes = detect_bubbles(input_image)
    balloons = 1
    for x,y,w,h in boxes.xywh.numpy():
        xmin, xmax = int(x - w/2), int(x + w/2)
        ymin, ymax = int(y - h/2), int(y + h/2)
        cropped_image = image[ymin:ymax,xmin:xmax]
        cropped_pil_image = Image.fromarray(cropped_image)
        generated_text = ocr_pipe(cropped_pil_image)[0]['generated_text']
        translated_text = translation_pipe(generated_text)[0]['translation_text']
        print(generated_text,translated_text)
        window_x,window_y,window_w,window_h = find_textbox(cropped_image)
        wrapped,font_size = get_font_size((window_w,window_h),translated_text,font_path)
        draw = ImageDraw.Draw(cropped_pil_image)
        draw.rectangle((window_x,window_y,window_x+window_w,window_y+window_h), fill = "white")
        font = ImageFont.truetype(font_path, font_size)

        x_center = int(window_w / 2) + window_x
        y_center = int(window_h/ 2) + window_y

        draw.text((x_center, y_center), wrapped, font=font, fill="black", anchor="mm",align='center')
        cropped_pil_image.save(f"temp/{balloons}c.png")
        image[ymin:ymax,xmin:xmax] = cv2.cvtColor(np.array(cropped_pil_image), cv2.COLOR_RGB2BGR)
        balloons += 1
    output = image
    cv2.imwrite(output_image,output)
    print(output_image)

