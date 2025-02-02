#!/bin/bash
# deploy.sh - Deployment script for QIP-X Framework

# Build the Docker image
IMAGE_NAME="qipx_framework"
docker build -t ${IMAGE_NAME} .

# Run the Docker container (detach mode)
docker run -d --name qipx_container -p 8080:8080 ${IMAGE_NAME}

# Tail the logs to monitor startup
docker logs -f qipx_container
