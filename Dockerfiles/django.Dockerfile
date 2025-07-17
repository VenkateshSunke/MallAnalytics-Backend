# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies including ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

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

# Make sure the entrypoint script has the correct permissions
RUN chmod +x /app/Dockerfiles/scripts/entrypoint.sh

RUN python manage.py collectstatic --noinput

# Add the command that was in docker-compose
ENTRYPOINT ["/bin/bash", "/app/Dockerfiles/scripts/entrypoint.sh"]

CMD ["gunicorn", "-c", "gunicorn_config.py", "--bind", "0.0.0.0:8000", "--workers", "2"]
