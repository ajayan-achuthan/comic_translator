import argparse
import os
import yaml
from translation.translator import Translator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manga Translator Tool.")
    parser.add_argument(
        "-l",
        "--languages",
        nargs="+",
        default=["jp", "en"],
        help="List of languages for translation.",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="Path to input file for OCR.",
    )
    parser.add_argument(
        "-o", "--output_file", type=str, help="Path to output file for translated text."
    )
    parser.add_argument(
        "--ocr_model", type=str, help="OCR model to use for text extraction."
    )
    parser.add_argument(
        "--translation_model",
        type=str,
        help="Translation model to use for language translation.",
    )

    args = parser.parse_args()

    # Load default models from config.yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    # Determine the default models based on the language pair
    language_pair = f"{args.languages[0]}_to_{args.languages[1]}"
    default_bubble_detector_model = (
        config["default_models"].get(language_pair, {}).get("bubble_detector")
    )
    default_ocr_model = config["default_models"].get(language_pair, {}).get("ocr")
    default_translation_model = (
        config["default_models"].get(language_pair, {}).get("translation")
    )

    # Use default models if not specified
    args.ocr_model = args.ocr_model or default_ocr_model
    args.translation_model = args.translation_model or default_translation_model

    # Create output file if not provided
    if args.output_file is None:
        file_name, file_ext = os.path.splitext(os.path.basename(args.input_file))
        args.output_file = os.path.join("output", f"{file_name}_converted.{file_ext}")

    translator = Translator(
        languages=args.languages,
        input_file=args.input_file,
        output_file=args.output_file,
        ocr_model=args.ocr_model,
        translation_model=args.translation_model,
        bubble_detector_model=default_bubble_detector_model,
    )
    translator.translate()
