import glob
import os
import random
import time

from evaluator import Evaluator
from pipelines import (ImprovedMultimodalOCREntityAnalysisPipeline,
                       ImprovedMultimodalOCRPipeline,
                       RawOCREntityAnalysisPipeline, RawOCRPipeline)


def verify():
    """Verify the OCR pipelines by testing them on a random image from the dataset.

    This function selects a random image from the SROIE2019 training set, loads the ground truth,
    and evaluates each pipeline's performance against it.
    """
    images = glob.glob("data/SROIE2019/train/img/*.jpg")
    if not images:
        print(
            "No images found. Please ensure the SROIE2019 dataset is downloaded and placed under `data/`."
        )
        return

    random.shuffle(images)
    img_path = images[0]
    print(f"Testing with {img_path}")
    print(f"LLM Used: {os.getenv('OLLAMA_LLM', 'qwen2.5:7b')}")
    print(f"VLM Used: {os.getenv('OLLAMA_VLM', 'llava:7b')}")

    dataset_dir = "data/SROIE2019/train"
    evaluator = Evaluator(dataset_dir)

    ground_truth = evaluator.load_ground_truth(img_path)
    print("Ground Truth Entities:", ground_truth["entities"])
    print("Ground Truth Raw Text Length:", len(ground_truth["ocr_text"]))

    pipelines = [
        RawOCRPipeline(),
        ImprovedMultimodalOCRPipeline(),
        RawOCREntityAnalysisPipeline(),
        ImprovedMultimodalOCREntityAnalysisPipeline(),
    ]

    for p in pipelines:
        print(f"\n--- Testing Pipeline: {p.__class__.__name__} ---")
        try:
            start_time = time.perf_counter()
            res = p.process(img_path)
            end_time = time.perf_counter()
            time_taken = end_time - start_time
            
            print("Raw Text Length:", len(res.get("raw_text", "")))
            print("Structured Data:", res.get("structured_data"))
            print(f"Time Taken: {time_taken:.2f} seconds")

            metrics = evaluator.evaluate(res, ground_truth)
            for metric, value in metrics.items():
                print(f"{metric}: {value}")
            print("=" * 40)

        except Exception as e:
            print(f"FAILED: {e}")


if __name__ == "__main__":
    verify()
