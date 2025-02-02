# Dockerfile for QIP-X Framework Deployment

FROM python:3.9-slim

# Install system-level dependencies
RUN apt-get update && apt-get install -y     build-essential     libssl-dev     libffi-dev     python3-dev     git     && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the QIP-X Framework source code
COPY . /app

# Expose any ports if needed
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run your main application script
CMD ["python", "main.py"]
