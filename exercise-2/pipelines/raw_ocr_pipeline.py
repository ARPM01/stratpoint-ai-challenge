from ocr_engine import OCREngine
from .base_pipeline import BasePipeline, PipelineOutput


class RawOCRPipeline(BasePipeline):
    """
    Scenario 1: Raw OCR
    Uses standard OCR (EasyOCR) to extract text.
    No entity extraction (returns raw text).
    """

    def __init__(self):
        """Initialize the RawOCRPipeline with an OCR engine."""
        self.ocr = OCREngine()

    def process(self, image_path: str) -> PipelineOutput:
        """Process an image using raw OCR to extract text.

        Args:
            image_path (str): The path to the image file to process.

        Returns:
            PipelineOutput: A dictionary with raw text, empty structured data, and pipeline name.
        """
        text = self.ocr.extract_text(image_path)
        return {
            "raw_text": text,
            "structured_data": {},  # No entity analysis
            "pipeline_name": "Raw OCR",
        }
