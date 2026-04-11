FROM python:3.11-slim

WORKDIR /app

# Install PromptForge dependencies (includes openenv-core)
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download DistilGPT-2 (perplexity guard) at build time to
# avoid cold-start delays on HF Spaces.
RUN python -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoTokenizer.from_pretrained('distilgpt2'); \
AutoModelForCausalLM.from_pretrained('distilgpt2'); \
print('distilgpt2 pre-cached')"

# Copy the full package
COPY . .

EXPOSE 7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
