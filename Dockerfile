# Visual Similarity Search: REST API (default) and Streamlit app.
#
#   docker compose up --build          # API on :8000, web app on :8501
#
# CPU-only, so it runs anywhere without a GPU. Images and the index are not
# baked in; they are mounted from ./data at run time.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app

# CPU builds of PyTorch are a fraction of the size of the default CUDA ones.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the CLIP weights at build time so containers start without a
# download and work offline. To use another model, rebuild with
#   --build-arg MODEL=... --build-arg PRETRAINED=...
# and index with the same model.
ARG MODEL=ViT-B-32
ARG PRETRAINED=laion2b_s34b_b79k
RUN python -c "import open_clip; open_clip.create_model_and_transforms('${MODEL}', pretrained='${PRETRAINED}')"

COPY src/ src/
COPY benchmarks/ benchmarks/

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data \
    && chown -R app /app/data /opt/huggingface
USER app

ENV INDEX_DIRECTORY=/app/data/index \
    HF_HUB_OFFLINE=1

EXPOSE 8000

# Healthy only once an index is loaded; /health returns 503 until then.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
