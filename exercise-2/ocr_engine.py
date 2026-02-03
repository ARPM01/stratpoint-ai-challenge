import easyocr


class OCREngine:
    """
    OCR Engine using EasyOCR to extract text from images.
    """

    def __init__(self, languages=["en"]):
        """
        Initializes the EasyOCR reader.

        Args:
            languages (List[str]): List of language codes for OCR.
        """
        self.reader = easyocr.Reader(languages, gpu=True)

    def extract_text(self, image_path: str) -> str:
        """
        Extracts raw text from an image path.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Extracted raw text.
        """
        try:
            result = self.reader.readtext(image_path, detail=0)
            return "\n".join(result)
        except Exception as e:
            print(f"Error during OCR: {e}")
            return ""

    def extract_text_with_boxes(self, image_path: str):
        """
        Extracts text with bounding boxes.

        Args:
            image_path (str): Path to the image file.

        Returns:
            List of tuples (bbox, text, conf)
        """
        try:
            result = self.reader.readtext(image_path)
            return result
        except Exception as e:
            print(f"Error during OCR: {e}")
            return []
