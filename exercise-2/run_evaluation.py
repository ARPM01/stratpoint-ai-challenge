import glob
import json
import os

import numpy as np
from evaluator import Evaluator
from pipelines import (
    ImprovedMultimodalOCREntityAnalysisPipeline,
    ImprovedMultimodalOCRPipeline,
    RawOCREntityAnalysisPipeline,
    RawOCRPipeline,
)
from tqdm import tqdm


def run_evaluation(num_samples=10):
    dataset_dir = "data/SROIE2019/train"
    images_dir = os.path.join(dataset_dir, "img")

    if not os.path.exists(images_dir):
        print(f"Error: Dataset directory not found at {images_dir}")
        return

    evaluator = Evaluator(dataset_dir)

    all_images = glob.glob(os.path.join(images_dir, "*.jpg"))
    if not all_images:
        print("No images found.")
        return

    # Select samples
    if num_samples and num_samples < len(all_images):
        test_images = all_images[:num_samples]
        print(f"Evaluating on {num_samples} random samples out of {len(all_images)}...")
    else:
        test_images = all_images
        print(f"Evaluating on all {len(all_images)} images...")

    # Define pipelines to evaluate
    pipelines = {
        "Raw OCR": RawOCRPipeline(),
        "Raw OCR + Entity": RawOCREntityAnalysisPipeline(),
        "Improved Multimodal OCR": ImprovedMultimodalOCRPipeline(),
        "Improved Multimodal OCR + Entity": ImprovedMultimodalOCREntityAnalysisPipeline(),
    }

    results = {}

    for name, pipeline in pipelines.items():
        print(f"\n--- Running Evaluation for: {name} ---")

        pipeline_metrics = {"ocr_cer": [], "ocr_wer": [], "entity_accuracy": []}

        for image_path in tqdm(test_images):
            try:
                # 1. Run Pipeline
                output = pipeline.process(image_path)

                # 2. Load Truth
                gt = evaluator.load_ground_truth(image_path)

                # 3. Evaluate
                metrics = evaluator.evaluate(output, gt)

                if "error" not in metrics:
                    if "ocr_cer" in metrics:
                        pipeline_metrics["ocr_cer"].append(metrics["ocr_cer"])
                    if "ocr_wer" in metrics:
                        pipeline_metrics["ocr_wer"].append(metrics["ocr_wer"])
                    if "entity_accuracy" in metrics:
                        pipeline_metrics["entity_accuracy"].append(
                            metrics["entity_accuracy"]
                        )

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        # Aggregate results
        avg_results = {}
        if pipeline_metrics["ocr_cer"]:
            avg_results["Average CER"] = np.mean(pipeline_metrics["ocr_cer"])
        if pipeline_metrics["ocr_wer"]:
            avg_results["Average WER"] = np.mean(pipeline_metrics["ocr_wer"])
        if pipeline_metrics["entity_accuracy"]:
            avg_results["Average Entity Accuracy"] = np.mean(
                pipeline_metrics["entity_accuracy"]
            )

        results[name] = avg_results

        print(f"Results for {name}:")
        print(json.dumps(avg_results, indent=2))

    print("\n\n=== Final Summary ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_evaluation(num_samples=5)
