import json
import os
from textwrap import dedent

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

load_dotenv(find_dotenv())


class RectificationAgent:
    def __init__(self):
        self.llm = ChatOllama(
            model=os.getenv("OLLAMA_LLM", "qwen2.5:7b"),
            base_url=os.getenv("OLLAMA_BASE_URL"),
            temperature=0,
        )

    def extract_entities(self, ocr_text):
        """
        Extracts structured data from OCR text using an LLM.
        """
        prompt = f"""
        Extract the following information from the receipt text below:
        1. Company Name (company)
        2. Date (date) in MM/DD/YYYY format
        3. Address (address)
        4. Total Amount (total) (Do not include currency symbols)

        Return the result as a Valid JSON object with keys: "company", "date", "address", "total".
        Do not include any other text or markdown formatting.

        Receipt Text:
        {ocr_text}
        """
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Simple cleanup to ensure we get JSON
            content = content.replace("```json", "").replace("```", "").strip()

            return json.loads(content)
        except Exception as e:
            print(f"Error extracting entities: {e}")

    def correct_ocr_text(self, ocr_text: str) -> str:
        """Correct and improve OCR-extracted text using an LLM.

        Args:
            ocr_text (str): Raw OCR-extracted text that may contain errors.

        Returns:
            str: Corrected text with improved formatting and readability.
        """
        prompt = dedent(
            f"""You are an expert at correcting OCR errors. Please correct the following OCR-extracted text, fixing spelling mistakes, improving formatting, and ensuring coherence. Return only the corrected text without any explanation.

OCR Text:
{ocr_text}"""
        )
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Error correcting OCR text: {e}")
            return ocr_text
