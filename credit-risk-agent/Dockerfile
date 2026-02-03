# Use Python 3.10 or 3.11 for better wheel compatibility
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir easyocr --no-deps

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
