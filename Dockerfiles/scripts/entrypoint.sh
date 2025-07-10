#!/bin/bash
set -e

echo "Starting entrypoint script..."

# Run migrations
echo "Running migrations..."
python manage.py migrate

# Start the main process
echo "Starting main process..."
exec "$@"