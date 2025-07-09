# Redis & Celery Setup Guide

This document provides a complete end-to-end setup guide for Redis and Celery integration in the Wise Backend project.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running the System](#running-the-system)
6. [Testing](#testing)
7. [API Endpoints](#api-endpoints)
8. [Monitoring](#monitoring)
9. [Troubleshooting](#troubleshooting)

## Overview

The system uses:
- **Redis** as a message broker and result backend for Celery
- **Celery** for asynchronous task processing
- **Django** with REST framework for API endpoints
- **PostgreSQL** for persistent data storage

### Architecture
```
Django App -> Redis Queue -> Celery Worker -> Database
     ↓             ↓              ↓
   REST API    Message Broker   Task Processing
```

## Prerequisites

1. **Python 3.8+** with pip
2. **Redis Server** (latest stable version)
3. **PostgreSQL** (for Django database)
4. **Virtual Environment** (recommended)

## Installation

### 1. Install Redis

#### macOS (using Homebrew):
```bash
brew install redis
brew services start redis
```

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

#### Windows:
Download and install from: https://redis.io/download

### 2. Install Python Dependencies

All required packages are already listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

Key packages include:
- `celery>=5.3.0`
- `redis>=4.5.0`
- `django`
- `djangorestframework`

## Configuration

### 1. Environment Variables

Create a `.env` file in your project root:
```env
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=  # Leave empty if no password

# Database Configuration (existing)
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=localhost
DB_PORT=5432
```

### 2. Django Settings

The following configurations are already added to `wise_backend/settings.py`:

```python
# Redis Configuration
REDIS_HOST = config('REDIS_HOST', default='localhost')
REDIS_PORT = config('REDIS_PORT', default=6379, cast=int)
REDIS_DB = config('REDIS_DB', default=0, cast=int)

# Celery Configuration
CELERY_BROKER_URL = f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'
CELERY_RESULT_BACKEND = CELERY_BROKER_URL
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
```

### 3. Apps Configuration

The logs app is added to `INSTALLED_APPS`:
```python
INSTALLED_APPS = [
    # ... other apps
    "wise_backend.logs",
]
```

## Running the System

### 1. Database Migration
```bash
python manage.py makemigrations
python manage.py migrate
```

### 2. Start Redis (if not running)
```bash
# macOS/Linux
redis-server

# Or as a service
brew services start redis  # macOS
sudo systemctl start redis-server  # Linux
```

### 3. Start Django Development Server
```bash
python manage.py runserver
```

### 4. Start Celery Worker (in a new terminal)
```bash
celery -A wise_backend worker -l info
```

### 5. Start Celery Beat (for periodic tasks, optional)
```bash
celery -A wise_backend beat -l info
```

## Testing

### 1. Using Management Commands

Test the entire setup:
```bash
python manage.py test_redis_celery --test-all
```

Test specific components:
```bash
# Test Redis only
python manage.py test_redis_celery --test-redis

# Test Celery only
python manage.py test_redis_celery --test-celery

# Add sample data to queue
python manage.py test_redis_celery --add-sample-data

# Process the queue
python manage.py test_redis_celery --process-queue
```

### 2. Using API Endpoints

Test system health:
```bash
curl http://localhost:8000/api/logs/health/
```

Test Celery:
```bash
curl http://localhost:8000/api/logs/test-celery/
```

Check queue status:
```bash
curl http://localhost:8000/api/logs/queue-status/
```

## API Endpoints

### Movement Log Management
- `POST /api/logs/add-movement/` - Add movement log to Redis queue
- `GET /api/logs/movements/` - Get movement logs with pagination

### Queue Management
- `GET /api/logs/queue-status/` - Get Redis queue status
- `POST /api/logs/process/` - Manually trigger log processing
- `POST /api/logs/clear-queue/` - Clear the movement logs queue

### System Monitoring
- `GET /api/logs/test-celery/` - Test Celery connection
- `GET /api/logs/health/` - Get system health status (Redis + Celery + DB)

### Blueprint Management (Core App)
- `GET /api/blueprints/` - List all blueprints
- `POST /api/blueprints/` - Create new blueprint
- `GET /api/blueprints/{id}/` - Get specific blueprint
- `PUT /api/blueprints/{id}/` - Update blueprint
- `DELETE /api/blueprints/{id}/` - Delete blueprint
- `POST /api/blueprints/{id}/apply/` - Apply blueprint to current mapping
- `POST /api/blueprints/create-from-current/` - Create blueprint from current mapping

## Monitoring

### 1. Celery Monitoring with Flower (Optional)

Install Flower:
```bash
pip install flower
```

Run Flower:
```bash
celery -A wise_backend flower
```

Access at: http://localhost:5555

### 2. Redis Monitoring

Redis CLI:
```bash
redis-cli
> INFO
> MONITOR
```

Check queue length:
```bash
redis-cli LLEN movement_logs
```

### 3. System Health Check

Check all components:
```bash
curl http://localhost:8000/api/logs/health/
```

Expected response:
```json
{
  "redis": true,
  "celery": true,
  "database": true,
  "overall_status": "healthy"
}
```

## Troubleshooting

### Common Issues

#### 1. Django Apps Not Loaded Error
**Error**: `django.core.exceptions.AppRegistryNotReady: Apps aren't loaded yet.`

**Solution**: The issue was in `wise_backend/celery.py` importing tasks at module level. Fixed by removing direct import and using autodiscover.

#### 2. Redis Connection Error
**Error**: `ConnectionError: Error 61 connecting to localhost:6379`

**Solutions**:
- Ensure Redis server is running: `redis-server`
- Check Redis configuration in `.env` file
- Test Redis connection: `redis-cli ping`

#### 3. Celery Worker Not Processing Tasks
**Error**: Tasks stay in PENDING state

**Solutions**:
- Ensure Celery worker is running: `celery -A wise_backend worker -l info`
- Check that broker URL is correct in settings
- Verify task is properly registered (check worker logs)

#### 4. Import Errors in Tasks
**Error**: `ModuleNotFoundError` when running tasks

**Solutions**:
- Ensure all apps are in `INSTALLED_APPS`
- Check Python path and virtual environment
- Restart Celery worker after code changes

### Debugging Commands

Check Redis connection:
```bash
python manage.py shell
>>> import redis
>>> from django.conf import settings
>>> client = redis.Redis(**settings.REDIS_CONFIG)
>>> client.ping()
```

Check Celery configuration:
```bash
python manage.py shell
>>> from wise_backend.celery import app
>>> print(app.conf)
```

Monitor Celery logs:
```bash
celery -A wise_backend worker -l debug
```

### Performance Tuning

#### Celery Worker Settings
```bash
# Multiple workers
celery -A wise_backend worker -l info --concurrency=4

# Different queue priorities
celery -A wise_backend worker -Q logs,core -l info
```

#### Redis Memory Optimization
Add to Redis configuration:
```
maxmemory 256mb
maxmemory-policy allkeys-lru
```

## Production Deployment

### 1. Use Production Redis Configuration
```python
# In production settings
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP = True
CELERY_BROKER_POOL_LIMIT = 10
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
```

### 2. Use Process Manager
Use supervisord or systemd to manage Celery workers:

```ini
# /etc/supervisor/conf.d/celery.conf
[program:celery]
command=celery -A wise_backend worker -l info
directory=/path/to/project
user=www-data
autostart=true
autorestart=true
```

### 3. Redis Security
- Set up Redis password
- Use Redis AUTH
- Configure firewall rules

## Example Usage

### Adding Movement Logs via API
```python
import requests

# Add a movement log
data = {
    "user_id": "USER123",
    "track_id": 1,
    "bbox": [100, 100, 200, 200],
    "state": "walking",
    "store": "Store A"
}

response = requests.post(
    "http://localhost:8000/api/logs/add-movement/", 
    json=data
)
print(response.json())
```

### Processing Logs Programmatically
```python
from wise_backend.logs.tasks import process_movement_logs_batch

# Process logs asynchronously
task = process_movement_logs_batch.delay()
result = task.get()  # Wait for completion
print(result)
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Celery and Redis logs
3. Use the management command to test individual components
4. Check system health endpoint

---

**Note**: This setup provides a robust foundation for processing movement logs asynchronously. The system can be extended to handle other types of background tasks as needed. 