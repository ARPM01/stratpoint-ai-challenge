from multimodal_agent import MultimodalAgent
from rectification_agent import RectificationAgent

from .base_pipeline import BasePipeline, PipelineOutput


class ImprovedMultimodalOCREntityAnalysisPipeline(BasePipeline):
    """
    Scenario 4: Improved OCR using VLM + entity analysis LLM
    Uses VLM for text, then LLM for entity extraction.
    """

    def __init__(self):
        self.vlm = MultimodalAgent()
        self.rectification_agent = RectificationAgent()

    def process(self, image_path: str) -> PipelineOutput:
        # Step 1: Improved OCR via VLM
        text = self.vlm.perform_ocr(image_path)

        # Step 2: Text correction via Rectification Agent
        corrected_text = self.rectification_agent.correct_ocr_text(text)

        # Step 3: Entity extraction on the corrected text
        entities = self.rectification_agent.extract_entities(corrected_text)

        return {
            "raw_text": corrected_text,
            "structured_data": entities,
            "pipeline_name": "Improved OCR (Multimodal) + Entity Analysis",
        }
