import base64
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

load_dotenv(find_dotenv())


class MultimodalAgent:
    def __init__(self):
        """
        Initializes the MultimodalAgent with a VLM (like LLaVA) via Ollama.
        """
        self.vlm = ChatOllama(
            model=os.getenv("OLLAMA_VLM", "llava:7b"),
            base_url=os.getenv("OLLAMA_BASE_URL"),
            temperature=0,
        )

    def perform_ocr(self, image_path: str) -> str:
        """
        Uses a Multimodal LLM (like LLaVA) via Ollama to extract text from an image.

        Args:
            image_path (str): Path to the input image.
        Returns:
            str: Extracted text from the image.
        """

        try:
            if not Path(image_path).exists():
                print(f"Error: Image file not found at {image_path}")
                return ""

            with open(image_path, "rb") as image_file:
                image_data = base64.standard_b64encode(image_file.read()).decode(
                    "utf-8"
                )

            message_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
                {
                    "type": "text",
                    "text": "Extract all visible text from this image. Return only the extracted text without any additional formatting or explanation.",
                },
            ]

            response = self.vlm.invoke([HumanMessage(content=message_content)])
            return response.content.strip()

        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return ""
        except Exception as e:
            print(f"Error calling VLM: {e}")
            return ""
