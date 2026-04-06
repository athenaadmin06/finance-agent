# Use Python 3.10 slim for better wheel compatibility
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# Removed build-essential since none of your packages need C-compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install dependencies using standard pip
# This will be fast because all versions in requirements.txt are now strictly pinned
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
