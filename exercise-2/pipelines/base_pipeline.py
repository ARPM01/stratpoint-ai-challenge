"""
Base pipeline class defining the interface for all OCR pipelines.
"""

from abc import ABC, abstractmethod
from typing import TypedDict, Dict, Any


class PipelineOutput(TypedDict):
    """TypedDict representing the output of an OCR pipeline."""

    raw_text: str
    structured_data: Dict[str, Any]
    pipeline_name: str


class BasePipeline(ABC):
    """Abstract base class for OCR pipelines.

    This class defines the interface that all OCR pipeline implementations must follow.
    """

    @abstractmethod
    def process(self, image_path: str) -> PipelineOutput:
        """Process an image and extract OCR data.

        Args:
            image_path (str): The path to the image file to process.

        Returns:
            PipelineOutput: A dictionary containing raw text, structured data, and pipeline name.
        """
        raise NotImplementedError
