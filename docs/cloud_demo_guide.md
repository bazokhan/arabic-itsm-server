# Cloud Demo Guide — Arabic ITSM Classifier Server

**Goal**: Run the demo server on a public URL so you can demo from anywhere (laptop, phone, supervisor's PC) without running a local server.

---

## The Key Insight: You Don't Need to Upload 600 MB

The trained encoder (`UBC-NLP/MARBERTv2`) is already on HuggingFace Hub.
You only need to upload the **custom heads** (`heads.pt` files) — typically 1–5 MB each.

```
Upload to HF Hub:
  heads.pt             ← 3 MB — your trained classification layers
  tokenizer files      ← already on Hub (not needed)

Server loads at runtime:
  UBC-NLP/MARBERTv2   ← 621 MB from Hub (cached after first run)
  heads.pt            ← from your uploaded repo
```

---

## Option A — HuggingFace Spaces (Recommended for Permanent Link)

**Best for**: Persistent demo URL you can share in your thesis.
**Cost**: Free (CPU). ~5–10 sec/inference on free tier.
**URL format**: `https://huggingface.co/spaces/<your-username>/<space-name>`

### Steps

**1. Upload your heads.pt files to HuggingFace Hub**

```bash
pip install huggingface_hub
huggingface-cli login    # uses your HF token

# Create a model repo per checkpoint (or put all in one Space repo)
huggingface-cli repo create arabic-itsm-l2 --type model
cd D:/AI/arabic-itsm-server/models/marbert_l2_best/
huggingface-cli upload arabic-itsm-l2 heads.pt heads.pt
```

**2. Create an HF Space**

- Go to huggingface.co → New Space → SDK: Docker → Hardware: CPU Basic (free)
- Clone the Space repo locally or use the web editor

**3. Add a Dockerfile to the server repo**

Create `Dockerfile` in `D:/AI/arabic-itsm-server/`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY . .
ENV MODEL_DIRS=models
ENV HF_HOME=/tmp/hf_cache
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

**4. Add a `.env` pointing to your HF model repos**

```env
# In the Space, models/ can contain lightweight stubs that load from HF Hub.
# The CheckpointClassifier will download heads.pt from HF if not local.
MODEL_DIRS=models
```

Or configure the server to load `CheckpointClassifier(model_path="your-hf-username/arabic-itsm-l2")` — the classifier already supports HF Hub paths.

**5. Push to the Space**

```bash
git add . && git commit -m "Deploy Arabic ITSM demo" && git push
```

HF Spaces builds automatically and gives you a public URL.

---

## Option B — Ngrok (Recommended for Defense Day Demo)

**Best for**: One-time demo from your local machine. Zero cloud setup.
**Cost**: Free
**URL**: Random subdomain like `https://abc123.ngrok.io` (changes each run)
**Requirement**: Your local machine must be on during the demo

### Steps

```bash
# 1. Install ngrok: https://ngrok.com/download
# 2. Start your server normally
uvicorn app.main:app --reload

# 3. In a second terminal, expose it
ngrok http 8000
# → Forwarding  https://abc123.ngrok-free.app → http://localhost:8000
```

Share the `https://...ngrok-free.app` link. Works from any device on any network.

**Tip**: With a free ngrok account you get a fixed subdomain — useful for a demo that spans multiple hours.

---

## Option C — Modal (Serverless ML Inference)

**Best for**: Pay-per-use, generous free tier (30 GPU-hours/month), no idle cost.
**Cost**: Free tier usually covers demo usage; ~$0.0004/second on GPU.
**URL**: Stable endpoint URL per deployment.

Modal is designed for ML workloads and handles model loading/caching automatically.

```bash
pip install modal
modal setup    # authenticate

# Create modal_app.py in the server repo
```

```python
# modal_app.py
import modal

app = modal.App("arabic-itsm-demo")
image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

@app.function(image=image, gpu="T4", timeout=300)
@modal.asgi_app()
def fastapi_app():
    from app.main import app
    return app
```

```bash
modal deploy modal_app.py
# → https://your-username--arabic-itsm-demo.modal.run
```

---

## Option D — Render (Simple GitHub Deploy)

**Best for**: Persistent URL, auto-deploys from GitHub, $7/month.
**Free tier**: Exists but only 512 MB RAM — too small for these models.

Paid "Starter" tier ($7/month) has 2 GB RAM which is sufficient for 1–2 loaded checkpoints.

### Steps

1. Push server repo to GitHub
2. Go to render.com → New Web Service → Connect GitHub repo
3. Set:
   - Build command: `pip install -r requirements.txt && pip install torch --index-url https://download.pytorch.org/whl/cpu`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Environment variables: `MODEL_DIRS=models`, `HF_HOME=/tmp/hf_cache`
4. Mount a disk for model storage or use HF Hub paths

---

## Which Option to Choose?

| Scenario | Recommendation |
|---|---|
| Thesis defense (one day) | **Option B (ngrok)** — 5 min setup, no accounts needed |
| Share with supervisor remotely | **Option B (ngrok)** — instant, free |
| Permanent link for thesis submission | **Option A (HF Spaces)** — free, professional |
| Production/research use | **Option C (Modal)** — serverless, GPU available |

---

## Model Loading on Cloud: Using HF Hub Paths

The `CheckpointClassifier` in `app/classifier.py` already supports HuggingFace Hub paths.
If you upload your `heads.pt` to a HF model repo, you can set:

```env
# .env for cloud deployment
MODEL_DIRS=
MULTITASK_MODEL_PATH=your-username/arabic-itsm-multitask
```

The classifier will download `heads.pt` from the Hub and load the encoder from `UBC-NLP/MARBERTv2` automatically.

**How to upload heads.pt to HF Hub:**

```bash
huggingface-cli login
huggingface-cli repo create arabic-itsm-multitask --type model
cd D:/AI/arabic-itsm-server/models/marbert_multi_task_best/
huggingface-cli upload your-username/arabic-itsm-multitask heads.pt heads.pt
```

---

## Storage Tip: Move the HuggingFace Cache Off C Drive

Add to your local `.env`:

```env
HF_HOME=D:/AI/cache/huggingface
```

This moves the ~600 MB MARBERTv2 download from `C:\Users\<you>\.cache\huggingface\` to `D:\AI\cache\`.
The setting takes effect before any model loading because `load_dotenv()` runs at server startup.
