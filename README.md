# LLM-based API Argument Completion with Knowledge-Augmented Prompts
This repository contains the code and resources necessary to reproduce the experimental results presented in the ICSE 2026 paper "LLM-based API Argument Completion with Knowledge-Augmented Prompts." Our paper introduces APICopilot, a novel approach that enhances LLM-based API argument completion using dynamically generated, context-rich prompts leveraging knowledge graphs and graph matching.
## Getting Started
### Prerequisites
* Python 3.8+
* Java Development Kit (JDK) 11+
* pip (Python package installer)
* Required Python libraries (install using `pip install -r requirements.txt`):
    ```
    antlr4-python3-runtime==4.9.2
    codellama==0.1.1
    networkx==2.8.8
    numpy==1.23.5
    openai==0.27.8
    torch==1.13.1
    transformers==4.25.1
* Access to the OpenAI API (for ChatGPT-4o)
* Access to the Google Gemini API (for Gemini 2.0 Flash)
* Hugging Face API token (if using Llama 3 from Hugging Face)
* Eclipse JDT (for Java code preprocessing)
* JavaParser library
## Datasets
This project uses the following datasets:
* **Eclipse and Netbeans projects (Java):** These datasets were originally used in [cite the PARC paper]. Instructions on how to obtain and preprocess this dataset can be found [Netbeans](https://github.com/apache/netbeans/tree/54987ffb73ae9e17b23d4a43a23770142f93206b), [Eclipse](https://www.eclipse.org/downloads/download.php?file=/eclipse/downloads/drops4/R-4.17-202009021800/eclipse-platform-sources-4.17.tar.xz).
* **PY150 dataset (Python):** This dataset is publicly available at [(https://huggingface.co/datasets/claudios/cubert_ETHPy150Open)]. 
* **Unseen Java Data:** The unseen Java code was collected from projects and repositories published after October 2023. The list of these projects and the scripts used for collection are available in the `data/unseen_data` directory.

Please ensure you have downloaded and preprocessed the datasets according to the provided instructions and place them in the appropriate directories as expected by the code.
