# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set the working directory
WORKDIR /app

# Install system dependencies
# We only need basic libs now, removed Tesseract/OpenCV system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure models directory exists
RUN mkdir -p models

# Expose port
EXPOSE 5000

# Default command runs the app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "main:app"]
