# Retina Disease Classification – FastAPI Inference Server

This repository provides a minimal FastAPI server to deploy your trained ViT model for retina OCT disease classification.

## Prepare the model

After training in your notebook/script, save artifacts to a directory, for example `my-trained-vit-model/`:

```python
model.save_pretrained('my-trained-vit-model')
processor.save_pretrained('my-trained-vit-model')
```

Ensure the directory contains files like `config.json`, `pytorch_model.bin` (or safetensors), `preprocessor_config.json`, etc.

## Install

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the API

```bash
set MODEL_DIR=my-trained-vit-model  # Windows
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000` to use the simple HTML upload form.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Predict with cURL:

```bash
curl -X POST "http://127.0.0.1:8000/predict?top_k=3" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.jpg"
```

Sample JSON response:

```json
{
  "predictions": [
    {"label": "CNV", "score": 0.91},
    {"label": "DME", "score": 0.06},
    {"label": "DRUSEN", "score": 0.02}
  ]
}
```

## Notes
- The server reads `MODEL_DIR` env var; defaults to `my-trained-vit-model` in the repo root.
- CPU inference by default. If you want GPU, move tensors to CUDA and ensure PyTorch with CUDA is installed.

