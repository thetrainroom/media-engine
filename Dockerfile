FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY polybos_engine/ polybos_engine/

# Install Python dependencies (CPU version by default)
RUN pip install --no-cache-dir -e ".[cpu]"

# Create directory for config
RUN mkdir -p /root/.config/polybos

EXPOSE 8000

CMD ["uvicorn", "polybos_engine.main:app", "--host", "0.0.0.0", "--port", "8000"]
