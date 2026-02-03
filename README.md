# stratpoint-ai-challenge

Code repository for the Stratpoint AI Challenge for Senior AI Engineer application.

## Overview

This repository contains solutions to the exercises provided in the Stratpoint AI Challenge. The challenge consists of two main exercises:

    - Challenge 1 - Predicting Solar Output (Data Science and LLMs),
    - Challenge 2 - Extracting Data From Receipts (Computer Vision, LLMs)

Install the required dependencies for both exercises using the provided `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

Modify the `.env` file to set the following environment variables:

    - `OLLAMA_BASE_URL`: The base URL for the Ollama server (e.g., `http://localhost:11434`).
    - `OLLAMA_LLM`: The LLM model to use for text processing (e.g., `qwen2.5:7b`).
    - `OLLAMA_VLM`: The VLM model to use for vision tasks (e.g., `llava:7b`).

## Challenge 1

TODO: Add info on jupyter notebook
TODO: Add info on how to run gradio demo

## Challenge 2

Place the SROIE2019 folder under the `exercise-2/data/` directory.

To test on a random image from the training set and evaluate the pipelines, run:

    ```bash
    python verify_pipeline.py
    ```

Run the Gradio app for receipt data extraction:
    ```bash
    cd exercise-2
    python app.py
    ```

## Disclaimer

This project was developed with significant assistance from an AI coding assistant. The majority of the code implementation was generated with LLM support, while the overall architecture, problem-solving approach, and final review were conducted by the author.
