# Use miniconda3 as base image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Add conda-forge channel
RUN conda config --add channels conda-forge

# Ensure all packages are installed from conda-forge
RUN conda config --set channel_priority strict

# Create conda environment
RUN conda create -y -n chatbot-gemini python pytorch torchvision numpy sentence-transformers faiss openai

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Activate conda environment and run the application
SHELL ["conda", "run", "-n", "chatbot-gemini", "/bin/bash", "-c"]
CMD ["python", "main.py"]
