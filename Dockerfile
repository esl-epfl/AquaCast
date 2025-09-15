# Pick a tag that matches your driver’s supported CUDA runtime
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
CMD ["bash", "./scripts/MyNN/synthesized.sh"]
