# LLM-based API Argument Completion with Knowledge-Augmented Prompts
This repository contains the code and resources necessary to reproduce the experimental results presented in the ICSE 2026 paper "LLM-based API Argument Completion with Knowledge-Augmented Prompts." Our paper introduces APICopilot, a novel approach that enhances LLM-based API argument completion using dynamically generated, context-rich prompts leveraging knowledge graphs and graph matching.
## Datasets
This project uses the following datasets:
* **Eclipse and Netbeans projects (Java):** These datasets were originally used in [cite the PARC paper]. Instructions on how to obtain and preprocess this dataset can be found [Netbeans](https://github.com/apache/netbeans/tree/54987ffb73ae9e17b23d4a43a23770142f93206b), [Eclipse](https://www.eclipse.org/downloads/download.php?file=/eclipse/downloads/drops4/R-4.17-202009021800/eclipse-platform-sources-4.17.tar.xz).
* **PY150 dataset (Python):** This dataset is publicly available at [(https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-token/dataset/py150)].
To use these dataset for code completion (API argument Completion task), run the preprocessing.py script to extract code files from the projects in this folder structure:
```text
APICopilot/
├── Dataset_Preprocessing/
│   ├── Eclipse_preprocessing.py
│   └── Netbeans_Preprocessing.py
│   └── PY150_Preprocessing.py
```
* **Unseen Java Data:** The unseen Java code was collected from projects and repositories published after October 2023, Due to size limits, the dataset has been uploaded to Google Drive. You can download the dataset using the following link:  [(https://drive.google.com/file/d/1QheSAfupFNCq_V4q4a4Mt8uHNDl_gpC2/view)].
The Unseen Java Data in this folder structure :
## Folder Structure
```plaintext
dataset/
│
│
└── Unseen Dataset
       ├── CLdata
       ├── file1.java
       ├── file2.java
       └── ...  
``` 

Please ensure you have downloaded and preprocessed the datasets according to the provided instructions and place them in the appropriate directories as expected by the code.

# APICopilot: Automated API Argument Completion Framework
![Overview of APICopilot](Main4.PNG)
## Class Hierarchy and Main Functions

### Class Hierarchy
```
[APICopilot.Main]
├── [Preprocessing (Eclipse, NetBeans, PY150)]
├── [ARExtractor]
├── [ExampleRetriever]
├── [KnowledgeTripleExtractor]
├── [KnowledgeGraphBuilder]
├── [GraphMatcher]
├── [PromptGenerator]
└── [ArgumentRecommender]
```

### Main Functions of Each Class

#### 1. **APICopilot**
- **`__init__(dataset_type, dataset_path, openai_api_key)`**: Initializes the framework with dataset and OpenAI API key.
- **`preprocess_dataset()`**: Preprocesses the dataset using the appropriate preprocessor.
- **`extract_argument_requests()`**: Extracts Argument Requests (ARs) from preprocessed data.
- **`retrieve_examples()`**: Retrieves similar examples for each AR.
- **`extract_knowledge_triples()`**: Extracts knowledge triples from ARs and examples.
- **`build_knowledge_graphs()`**: Builds knowledge graphs from knowledge triples.
- **`perform_graph_matching()`**: Performs graph matching to find similar subgraphs.
- **`generate_prompts()`**: Generates prompts for LLM-based argument completion.
- **`recommend_arguments()`**: Recommends arguments using LLM-based prediction.
- **`run_pipeline()`**: Runs the full APICopilot pipeline.

#### 2. **EclipsePreprocessing**
- **`preprocess()`**: Preprocesses Eclipse Java files.

#### 3. **NetBeansPreprocessing**
- **`preprocess()`**: Preprocesses NetBeans Java files.

#### 4. **PY150Preprocessing**
- **`preprocess()`**: Preprocesses PY150 Python files.

#### 5. **ARExtractor**
- **`extract_ar(preprocessed_data)`**: Extracts Argument Requests (ARs) from preprocessed data.

#### 6. **ExampleRetriever**
- **`retrieve_examples(ar)`**: Retrieves similar examples for a given AR.

#### 7. **KnowledgeTripleExtractor**
- **`extract_triples(ar)`**: Extracts knowledge triples from an AR.

#### 8. **KnowledgeGraphBuilder**
- **`build_g_input(ar_triples)`**: Builds the input knowledge graph.
- **`build_kg_examples(example_triples)`**: Builds the example knowledge graph.

#### 9. **GraphMatcher**
- **`find_isomorphic_subgraphs(kg_input, kg_examples)`**: Finds isomorphic subgraphs in the knowledge graphs.

#### 10. **PromptGenerator**
- **`generate_prompt(ar, matched_subgraphs)`**: Generates a prompt for LLM-based argument completion.

#### 11. **ArgumentRecommender**
- **`recommend_arguments(prompt)`**: Recommends arguments using LLM-based prediction.

## Flow of Information Between Objects

1. **Preprocessing**:
   - `APICopilot` initializes the appropriate preprocessor (`EclipsePreprocessing`, `NetBeansPreprocessing`, or `PY150Preprocessing`).
   - The preprocessor preprocesses the dataset and returns preprocessed data.

2. **Argument Extraction**:
   - `APICopilot` uses `ARExtractor` to extract Argument Requests (ARs) from the preprocessed data.

3. **Example Retrieval**:
   - `APICopilot` uses `ExampleRetriever` to retrieve similar examples for each AR.

4. **Knowledge Triple Extraction**:
   - `APICopilot` uses `KnowledgeTripleExtractor` to extract knowledge triples from ARs and examples.

5. **Knowledge Graph Construction**:
   - `APICopilot` uses `KnowledgeGraphBuilder` to build knowledge graphs from the extracted triples.

6. **Graph Matching**:
   - `APICopilot` uses `GraphMatcher` to find isomorphic subgraphs in the knowledge graphs.

7. **Prompt Generation**:
   - `APICopilot` uses `PromptGenerator` to generate prompts for LLM-based argument completion.

8. **Argument Recommendation**:
   - `APICopilot` uses `ArgumentRecommender` to recommend arguments using LLM-based prediction.

## Detailed Workflow

1. **Dataset Preprocessing**:
   - The dataset is preprocessed to clean and normalize the code files.
   - The preprocessed data is passed to the `ARExtractor`.

2. **Argument Request Extraction**:
   - The `ARExtractor` identifies method calls and extracts arguments.
   - The extracted ARs are passed to the `ExampleRetriever`.

3. **Example Retrieval**:
   - The `ExampleRetriever` retrieves similar examples for each AR.
   - The retrieved examples are passed to the `KnowledgeTripleExtractor`.

4. **Knowledge Triple Extraction**:
   - The `KnowledgeTripleExtractor` extracts knowledge triples from ARs and examples.
   - The extracted triples are passed to the `KnowledgeGraphBuilder`.

5. **Knowledge Graph Construction**:
   - The `KnowledgeGraphBuilder` constructs knowledge graphs from the triples.
   - The constructed graphs are passed to the `GraphMatcher`.

6. **Graph Matching**:
   - The `GraphMatcher` finds isomorphic subgraphs in the knowledge graphs.
   - The matched subgraphs are passed to the `PromptGenerator`.

7. **Prompt Generation**:
   - The `PromptGenerator` creates context-rich prompts for LLM-based argument completion.
   - The generated prompts are passed to the `ArgumentRecommender`.

8. **Argument Recommendation**:
   - The `ArgumentRecommender` uses the LLM to predict missing arguments.
   - The recommended arguments are returned to the `APICopilot` for final output.

## Example Usage

```python
# Initialize APICopilot with Eclipse dataset
api_copilot = APICopilot(
    dataset_type="eclipse",
    dataset_path="/path/to/eclipse/project",
    openai_api_key="your-openai-api-key"
)

# Run the full pipeline
api_copilot.run_pipeline()
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
### Example Usage
```
api_copilot = APICopilot(
    dataset_type="eclipse",
    dataset_path="/path/to/eclipse/project",
    openai_api_key="your-openai-api-key"
)
api_copilot.run_pipeline()
```
