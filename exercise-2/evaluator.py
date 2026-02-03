import json
import os

from jiwer import Compose, RemovePunctuation, Strip, ToLowerCase, cer, wer


class Evaluator:
    def __init__(self, dataset_dir):
        """
        dataset_dir: path to 'train' directory, containing 'box' and 'entities' folders.
        """
        self.dataset_dir = dataset_dir
        self.box_dir = os.path.join(dataset_dir, "box")
        self.entities_dir = os.path.join(dataset_dir, "entities")

    def load_ground_truth(self, filename):
        """
        Loads both OCR text and Entity data for a given filename (e.g., X0001.jpg/txt).
        Returns a dict with 'ocr_text' and 'entities'.
        """
        basename = os.path.basename(filename)
        txt_name = os.path.splitext(basename)[0] + ".txt"

        ocr_gt = self._load_ocr_gt(txt_name)
        entities_gt = self._load_entities_gt(txt_name)

        return {"ocr_text": ocr_gt, "entities": entities_gt}

    def _load_ocr_gt(self, txt_name):
        path = os.path.join(self.box_dir, txt_name)
        full_text = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    # Format: x1,y1,x2,y2,x3,y3,x4,y4,TEXT\n
                    parts = line.strip().split(",", 8)
                    if len(parts) > 8:
                        full_text.append(parts[8])
        return "\n".join(full_text)

    def _load_entities_gt(self, txt_name):
        path = os.path.join(self.entities_dir, txt_name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def evaluate(self, pipeline_result, ground_truth):
        """
        Compares pipeline output with ground truth.
        """
        if not ground_truth or (
            not ground_truth["ocr_text"] and not ground_truth["entities"]
        ):
            return {"error": "No ground truth found"}

        metrics = {}

        # 1. Evaluate OCR (if GT exists)
        if ground_truth["ocr_text"]:
            gt_text = ground_truth["ocr_text"]
            pred_text = pipeline_result.get("raw_text", "")

            try:
                metrics["ocr_cer"] = cer(
                    gt_text,
                    pred_text,
                )
                metrics["ocr_wer"] = wer(
                    gt_text,
                    pred_text,
                )
            except Exception as e:
                metrics["ocr_error"] = str(e)

        # 2. Evaluate Entities (if GT exists and pipeline produced data)
        if ground_truth["entities"] and pipeline_result.get("structured_data"):
            gt_entities = ground_truth["entities"]
            pred_entities = pipeline_result.get("structured_data", {})

            correct_fields = 0
            total_fields = 0

            for key in ["company", "date", "address", "total"]:
                if key in gt_entities and key in pred_entities:
                    # Basic normalization
                    gt_val = str(gt_entities[key]).strip().lower()
                    pred_val = str(pred_entities[key]).strip().lower()

                    if gt_val == pred_val:
                        correct_fields += 1
                total_fields += 1

            metrics["entity_accuracy"] = (
                correct_fields / total_fields if total_fields > 0 else 0
            )

        return metrics


if __name__ == "__main__":
    # Simple test
    evaluator = Evaluator("data/SROIE2019/train")
    gt = evaluator.load_ground_truth("data/SROIE2019/train/img/X51008145450.jpg")
    print("Loaded Ground Truth:", gt)
    sample_result = {
        "raw_text": gt["ocr_text"],
        "structured_data": gt["entities"],
    }
    sample_metrics = evaluator.evaluate(sample_result, gt)
    print("Evaluation Metrics:", sample_metrics)
