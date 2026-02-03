from multimodal_agent import MultimodalAgent
from .base_pipeline import BasePipeline, PipelineOutput


class ImprovedMultimodalOCRPipeline(BasePipeline):
    """
    Scenario 2: Improved OCR using multimodal LLM
    Uses a VLM to transcribe text from the image.
    No entity extraction (returns raw text).
    """

    def __init__(self):
        self.vlm = MultimodalAgent()

    def process(self, image_path: str) -> PipelineOutput:
        text = self.vlm.perform_ocr(image_path)
        return {
            "raw_text": text,
            "structured_data": {},  # No entity analysis
            "pipeline_name": "Improved OCR (Multimodal)",
        }
