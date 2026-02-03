from ocr_engine import OCREngine
from rectification_agent import RectificationAgent

from .base_pipeline import BasePipeline, PipelineOutput


class RawOCREntityAnalysisPipeline(BasePipeline):
    """
    Scenario 3: Raw OCR + entity analysis LLM
    Uses EasyOCR for text, then LLM for entity extraction.
    """

    def __init__(self):
        """Initialize the RawOCREntityAnalysisPipeline with OCR engine and entity agent."""
        self.ocr = OCREngine()
        self.rectification_agent = RectificationAgent()

    def process(self, image_path: str) -> PipelineOutput:
        """Process an image using raw OCR and LLM-based entity analysis.

        Args:
            image_path (str): The path to the image file to process.

        Returns:
            PipelineOutput: A dictionary with raw text, extracted entities, and pipeline name.
        """
        text = self.ocr.extract_text(image_path)
        entities = self.rectification_agent.extract_entities(text)

        return {
            "raw_text": text,
            "structured_data": entities,
            "pipeline_name": "Raw OCR + Entity Analysis",
        }
