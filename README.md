## Content Classification Using LLM
content classificiation is an important task across various industries. In this repository, I explore content classification using 
using quantized Large Language models (LLMs) and LLama-cpp. The goals are as follows:
* Perform content classifications on
* Only CPU for the task.
* Strict JSON responses with minimum variabilty.
* Assess the quality of the prediction.
* Find a model that a gives right balance of speed and accuracy.
* Check/Fix Hallucinations.

## Getting Started
I use the following environment:
* Mac M1 pro (sonoma, 16GB RAM)
* Python 3.9.9

To get start, create a virtual environment:
```python
python -m venv env
source env/bin/activate
```
Install the requirements:

```python
pip install -r requirements.txt
```
Note: if there is an issue with llama-cpp-python installation, try the following:
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install -U "llama-cpp-python[server]==0.2.56" --no-cache-dir
```

To download the model you need, `huggingface-cli`. You can download `Yarn-Mistral-7B-128k-GGUF` model as follows:

```bash
huggingface-cli download TheBloke/Yarn-Mistral-7B-128k-GGUF yarn-mistral-7b-128k.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
```


## Example usage

```python
import json
from content_classification.classifier import LLMClassifier

classification_config = {
    "model_path": "model/yarn-mistral-7b-128k.Q4_K_M.gguf",
    "IAB_categories": True,
    "Gender": True,
    "Topics": True,
    "struct":
        {
        "IAB_categories": ["IAB1", "IAB2", "IAB3"],
        "Age_groups": ["Adult", "Teen"],
        "Topics": ["Topic1", "Topic2"],
        "Gender": ["Male"]
        }
    }

  content = """Villa living embodies a unique blend of luxury, comfort, and tranquility, offering a haven away from the hustle and bustle of city life. Nestled amidst serene landscapes or perched on scenic coastal cliffs, villas beckon those seeking a retreat where relaxation and rejuvenation are paramount"""

  classifier = LLMClassifier(classification_config)
  response = classifier.get_response(content)
  print(json.dumps(response, indent=2))
```
The output is as follows:

```json
{
  "IAB_categories": [
    "Lifestyle",
    "Travel",
    "Real Estate",
    "Home & Garden"
  ],
  "Topics": [
    "Luxury",
    "Comfort",
    "Tranquility",
    "Relaxation"
  ],
  "Gender": [
    "Male"
  ]
}
```

## Serving the response via LLAMA-cpp Rest-API
To receive responses from the API LLAMA-cpp Rest-API, run the bash script start_endpoints.sh

```bash
bash start_endpoints.sh
```
You should see something like this:

```bash
Model metadata: {'general.quantization_version': '2', 'tokenizer.ggml.unknown_token_id': '0', 'tokenizer.ggml.eos_token_id': '2', 'tokenizer.ggml.bos_token_id': '1', 'tokenizer.ggml.model': 'llama', 'llama.rope.scaling.original_context_length': '8192', 'llama.rope.scaling.factor': '16.000000', 'llama.rope.scaling.type': 'yarn', 'llama.attention.head_count_kv': '8', 'llama.context_length': '131072', 'llama.attention.head_count': '32', 'llama.rope.freq_base': '10000.000000', 'llama.rope.dimension_count': '128', 'general.file_type': '15', 'llama.rope.scaling.finetuned': 'true', 'llama.feed_forward_length': '14336', 'llama.embedding_length': '4096', 'llama.block_count': '32', 'general.architecture': 'llama', 'llama.attention.layer_norm_rms_epsilon': '0.000010', 'general.name': 'nousresearch_yarn-mistral-7b-128k'}
INFO:     Started server process [33544]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)
```

Now, just use `APIClassifier` class with the similar structure:

```python
from content_classification.classifier import APIClassifier

classification_config = {
    "model_path": "model/yarn-mistral-7b-128k.Q4_K_M.gguf",
    "IAB_categories": True,
    "Gender": True,
    "Topics": True,
    "struct": {
        "IAB_categories": ["IAB1", "IAB2", "IAB3"],
        "Age_groups": ["Adult", "Teen"],
        "Topics": ["Topic1", "Topic2"],
        "Gender": ["Male"],
    },
}


content = """Villa living embodies a unique blend of luxury, comfort, and tranquility, offering a haven 
away from the hustle and bustle of city life. Nestled amidst serene landscapes or perched on scenic coastal cliffs, 
villas beckon those seeking a retreat where relaxation and rejuvenation are paramount"""

classifier = APIClassifier(classification_config)
response = classifier.get_response(content)
print(response)
```

