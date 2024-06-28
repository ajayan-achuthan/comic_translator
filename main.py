import os
import argparse

from translator import translator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This Manga Translator is a tool that utilizes various resources and models to extract text from manga images, translate the text. The following libraries and models are used in this translator:')
    
    parser.add_argument('-l', '--languages', nargs='+', choices=[['jp', 'en'], ['en', 'ml']], default=['jp', 'en'], help='List of languages for translation (e.g. en es). Default is English to Spanish.')
    parser.add_argument('-i', '--input_file', type=str, required=True, help='Path to input file for OCR.')
    parser.add_argument('-o', '--output_file', type=str, help='Path to output file for translated text.')
    parser.add_argument('--ocr_model', type=str, default='default', help='OCR model to use for text extraction. Default is “default”.')
    parser.add_argument('--translator_model', type=str, default='models/Helsinki-NLP/opus-mt-ja-en', help='Translator model to use for language translation. Default is “default”.')

    args = parser.parse_args()
    
    if args.output_file is None:
        file_name, file_ext = args.input_file.rsplit('.', 1)
        args.output_file = f'{file_name}_converted.{file_ext}'
    if args.languages == ['jp', 'en']:
        #japanese
        if args.ocr_model == 'default':
            ocr_pipeline = "models/kha-white/manga-ocr-base"
            translation_pipe = "models/Helsinki-NLP/opus-mt-ja-en"
    translator_pipeline = translator(languages = args.languages, 
                                     input_file = args.input_file, 
                                     output_file = args.output_file, 
                                     ocr_pipename = ocr_pipeline, 
                                     translation_pipename = translation_pipe)
    translator_pipeline.translate()
