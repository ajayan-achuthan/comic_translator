# Manga Translator

This Manga Translator is a tool that utilizes various resources and models to extract text from manga images, translate the text. The following libraries and models are used in this translator:

- OCR Models:
- chatgpt
- kha-white/manga-ocr-base
- pytesseract
- spacy

- Translator Models:
- chatgpt
- Helsinki-NLP/opus-mt-ja-en

- Bubble Detection:
- YOLO8

## Usage

To use the Manga Translator, follow these steps:

1. Install the necessary libraries and models mentioned above.
2. Run the Manga Translator CLI app with the desired input options:

    python app.py -l jp en -i manga_page.png --ocr_model chatgpt --translator_model chatgpt

    Here, the -l flag specifies the languages for translation, -i specifies the input manga image, --ocr_model specifies the OCR model to use, and --translator_model specifies the translator model to use.

3. View the translated text and bubble detection results in the output file.

Feel free to explore other available options and models in the Manga Translator for your manga translation needs. Enjoy translating manga effortlessly!