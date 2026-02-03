import glob
import json
import os

import gradio as gr
from evaluator import Evaluator
from pipelines import (
    ImprovedMultimodalOCREntityAnalysisPipeline,
    ImprovedMultimodalOCRPipeline,
    RawOCREntityAnalysisPipeline,
    RawOCRPipeline,
)

# Paths
DATASET_PATH = "data/SROIE2019"  # Relative to exercise-2
TRAIN_IMAGES_PATH = os.path.join(DATASET_PATH, "train", "img")

# Initialize Pipelines
pipelines = {
    "Raw OCR": RawOCRPipeline(),
    "Improved OCR (Multimodal)": ImprovedMultimodalOCRPipeline(),
    "Raw OCR + Entity Analysis": RawOCREntityAnalysisPipeline(),
    "Improved OCR (Multimodal) + Entity Analysis": ImprovedMultimodalOCREntityAnalysisPipeline(),
}
evaluator = Evaluator(os.path.join(DATASET_PATH, "train"))


def process_pipeline(image_selection, pipeline_choice, run_evaluation):
    # Handle Radio selection - returns the file path directly
    image_path = image_selection

    if not image_path:
        return "Error: No image selected", "", "{}", "{}", "{}"

    if pipeline_choice not in pipelines:
        return "Error: Invalid Pipeline", "", "{}", "{}", "{}"

    pipeline = pipelines[pipeline_choice]

    # Run the pipeline
    result = pipeline.process(image_path)

    raw_text = result.get("raw_text", "")
    features = result.get("structured_data", {})

    # Load ground truth
    gt = evaluator.load_ground_truth(image_path)
    gt_text = gt.get("ocr_text", "Ground Truth not available")
    gt_entities = gt.get("entities", {})

    # Evaluation
    eval_result = "Ground Truth not found or Evaluation disabled."
    if run_evaluation:
        if gt["ocr_text"] or gt["entities"]:
            metrics = evaluator.evaluate(result, gt)
            eval_result = json.dumps(metrics, indent=2)
        else:
            eval_result = "No Ground Truth file found for this image (checked 'box' and 'entities' folders)."

    return (
        raw_text,
        gt_text,
        json.dumps(features, indent=2),
        json.dumps(gt_entities, indent=2),
        eval_result,
    )


def list_sample_images():
    if os.path.exists(TRAIN_IMAGES_PATH):
        return glob.glob(os.path.join(TRAIN_IMAGES_PATH, "*.jpg"))[:3]
    return []


# Gradio Interface
with gr.Blocks(title="SROIE OCR Prototype") as demo:
    gr.Markdown("# SROIE OCR")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Select a Receipt")
            samples = list_sample_images()
            input_image = gr.Radio(
                choices=[(os.path.basename(s), s) for s in samples],
                value=samples[0] if samples else None,
                label="Select an image to process",
                type="value",
            )
            image_gallery = gr.Gallery(
                value=[samples[0]] if samples else [],
                label="Selected Receipt",
                columns=1,
                rows=1,
                height="auto",
                object_fit="contain",
                interactive=False,
            )
            pipeline_dropdown = gr.Dropdown(
                choices=list(pipelines.keys()),
                value="Raw OCR + Entity Analysis",
                label="Select OCR/Extraction Pipeline",
            )
            run_eval_checkbox = gr.Checkbox(
                label="Run Evaluation (Compare with GT)", value=True
            )
            submit_btn = gr.Button("Process Receipt")

        with gr.Column():
            with gr.Row():
                output_raw = gr.Textbox(label="Raw OCR Text", lines=10)
                output_gt = gr.Textbox(label="Ground Truth Text", lines=10)
            with gr.Row():
                output_json = gr.Code(label="Extracted Entities", language="json")
                output_gt_entities = gr.Code(
                    label="Ground Truth Entities", language="json"
                )
            output_eval = gr.Code(label="Evaluation Metrics", language="json")

    # Update gallery when radio selection changes
    input_image.change(
        fn=lambda x: [x] if x else [], inputs=[input_image], outputs=[image_gallery]
    )

    submit_btn.click(
        fn=process_pipeline,
        inputs=[input_image, pipeline_dropdown, run_eval_checkbox],
        outputs=[output_raw, output_gt, output_json, output_gt_entities, output_eval],
    )

if __name__ == "__main__":
    demo.launch(share=False)
