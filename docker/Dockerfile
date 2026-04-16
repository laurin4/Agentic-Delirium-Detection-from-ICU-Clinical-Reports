FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal system packages used by matplotlib and common IO backends
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libfreetype6 \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Runtime writable dirs
RUN mkdir -p /app/outputs /app/Data/Raw

# Default interactive shell; run pipeline modules explicitly via docker run
CMD ["bash"]
