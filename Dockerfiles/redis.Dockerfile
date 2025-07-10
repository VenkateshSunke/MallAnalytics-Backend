FROM bitnami/redis:7.0.15

# Switch to root to set permissions
USER root

# Ensure correct permissions for Redis data directory
RUN mkdir -p /redis/data && chown -R 1001:1001 /redis/data

# Switch back to the Redis user
USER 1001

# Use the default entrypoint
ENTRYPOINT ["/opt/bitnami/scripts/redis/entrypoint.sh"]

# Command to run the Redis server with the config
CMD ["redis-server", "/etc/redis/redis.conf"]
