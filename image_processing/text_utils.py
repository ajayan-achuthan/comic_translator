import cv2
import matplotlib.pyplot as plt
from PIL import ImageFont
from string import ascii_letters
import textwrap


def find_textbox(image):
    # Load image, grayscale, Gaussian blur, adaptive threshold
    # image = cv2.imread('temp/temp/1.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30
    )

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    ROI_number = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 10000:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
            # cv2.imshow('d',image)
            # ROI = image[y:y+h, x:x+w]
            # cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
            # ROI_number += 1
            return cv2.boundingRect(c)
    return None


# from gpt
# def find_textbox(image):
#     try:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30)
#         cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
#         for c in cnts:
#             if cv2.contourArea(c) > 10000:
#                 return cv2.boundingRect(c)
#     except Exception as e:
#         print(f"Error finding text box: {e}")
#     return None


def get_font_size(textarea, text, font_name, pixel_gap=11):
    text_width = int(textarea[0])
    text_height = int(textarea[1])

    for point_size in range(5, 90):
        wrapped_text = []
        font = ImageFont.truetype(font_name, point_size)

        avg_char_width = sum(font.getbbox(char)[2] for char in ascii_letters) / len(
            ascii_letters
        )
        max_char_height = max(
            font.getbbox(char)[3] - font.getbbox(char)[1] for char in ascii_letters
        )

        # Translate this average length into a character count
        max_char_count = int((text_width) / avg_char_width)
        text = textwrap.fill(text=text, width=max_char_count)
        num_line = len(text.splitlines())

        wrapped_text.append(text)

        if (max_char_height * num_line) + (pixel_gap * (num_line + 1)) >= text_height:
            point_size = point_size - 1
            text = wrapped_text[-1]

            # print("\n --> SIZE: ", point_size)
            break

    return text, point_size
