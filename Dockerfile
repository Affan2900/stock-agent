FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Install the CPU-only torch build first. The default PyPI wheel pulls ~2.5GB of
# CUDA/nvidia dependencies that are dead weight on GPU-less nodes. Installing it
# up front means the requirements.txt torch pin is already satisfied below.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501 8502

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
