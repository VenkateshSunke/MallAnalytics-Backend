# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install debugpy

# Copy the current directory contents into the container at /app
COPY . /app/

# Default command will be overridden by task definition
CMD ["celery", "-A", "wise_backend.celery", "worker", "-l", "info", "-E", "--pool=prefork", "--autoscale=10,2", "-Q", "default,qws"]