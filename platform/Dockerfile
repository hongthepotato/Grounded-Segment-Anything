FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

COPY platform/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy platform source
COPY platform/ /app/platform/

# Copy blocks catalog (block.json files only — not full Dockerfiles, saves image size)
COPY blocks/ /app/blocks/

# Graph storage (bind-mounted at runtime for persistence)
RUN mkdir -p /app/graphs /app/data

# Auth file location (bind-mounted for persistence)
ENV AUTH_FILE_PATH=/data/auth.json

EXPOSE 8000

CMD ["uvicorn", "platform.app:app", "--host", "0.0.0.0", "--port", "8000"]
