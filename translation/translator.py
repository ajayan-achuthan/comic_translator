import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path
from transformers import pipeline
from detection.bubble_detector import BubbleDetector
from image_processing.text_utils import find_textbox, get_font_size


class OCRPipeline:
    def __init__(self, model_name, ocr_type="transformers"):
        self.ocr_type = ocr_type
        if ocr_type == "easyocr":
            import easyocr

            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            self.ocr_pipeline = easyocr.Reader(["en"])
        elif ocr_type == "transformers":
            self.ocr_pipeline = pipeline("image-to-text", model=model_name)
        else:
            raise ValueError("Unsupported OCR type. Use 'easyocr' or 'transformers'.")

    def extract_text(self, image):
        try:
            if self.ocr_type == "easyocr":
                result = self.ocr_pipeline.readtext(image)
                return " ".join([text for (_, text, _) in result])
            elif self.ocr_type == "transformers":
                image = Image.fromarray(image)
                return self.ocr_pipeline(image)[0]["generated_text"]
        except Exception as e:
            print(f"Error during text extraction: {e}")
            return ""


class TranslationPipeline:
    def __init__(self, model_name):
        self.translation_pipeline = pipeline("translation", model=model_name)

    def translate_text(self, text):
        return self.translation_pipeline(text)[0]["translation_text"]


class Translator:
    def __init__(
        self,
        languages,
        input_file,
        output_file,
        ocr_model,
        translation_model,
        bubble_detector_model,
    ):
        self.languages = languages
        self.input_file = input_file
        self.output_file = output_file
        self.bubble_detector = BubbleDetector(bubble_detector_model)
        self.ocr_pipeline = OCRPipeline(ocr_model)
        self.translation_pipeline = TranslationPipeline(translation_model)
        self.font_path = "assets/manga.ttf"

    def translate_image(self, image):
        boxes = self.bubble_detector.detect_bubbles(image)
        balloons = 1
        for x, y, w, h in boxes.xywh.numpy():
            xmin, xmax = int(x - w / 2), int(x + w / 2)
            ymin, ymax = int(y - h / 2), int(y + h / 2)
            cropped_image = image[ymin:ymax, xmin:xmax]
            cropped_pil_image = Image.fromarray(cropped_image)
            generated_text = self.ocr_pipeline.extract_text(cropped_image)
            translated_text = self.translation_pipeline.translate_text(generated_text)
            window_x, window_y, window_w, window_h = find_textbox(cropped_image)
            wrapped, font_size = get_font_size(
                (window_w, window_h), translated_text, self.font_path
            )
            draw = ImageDraw.Draw(cropped_pil_image)
            draw.rectangle(
                (window_x, window_y, window_x + window_w, window_y + window_h),
                fill="white",
            )
            font = ImageFont.truetype(self.font_path, font_size)
            x_center = int(window_w / 2) + window_x
            y_center = int(window_h / 2) + window_y
            draw.text(
                (x_center, y_center),
                wrapped,
                font=font,
                fill="black",
                anchor="mm",
                align="center",
            )
            image[ymin:ymax, xmin:xmax] = cv2.cvtColor(
                np.array(cropped_pil_image), cv2.COLOR_RGB2BGR
            )
            balloons += 1

        return image

    def translate(self):
        input_name, input_ext = os.path.splitext(self.input_file)
        if input_ext.lower() == ".pdf":
            images = convert_from_path(self.input_file)
            im1 = []
            for i, image in enumerate(images):
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                res_image = self.translate_image(image)
                res_image = Image.fromarray(res_image)
                pdf_bytes = res_image.convert("RGB")
                im1.append(pdf_bytes)
            im1[0].save(
                f"{input_name}_convered.pdf", save_all=True, append_images=im1[1:]
            )
        elif input_ext.lower() in [".png", ".jpg", ".jpeg"]:
            # Convert Image and rotate
            image = cv2.imread(self.input_file)
            res_image = self.translate_image(image)
            cv2.imwrite(self.output_file, res_image)

            print(f"Image translated successfully. Saved as {self.output_file}")
        else:
            print("File should be png, jpeg, or pdf")
