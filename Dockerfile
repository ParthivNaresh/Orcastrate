# Production Dockerfile for Orcastrate
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r orcastrate && useradd -r -g orcastrate orcastrate

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-dev

# Copy application code
COPY src/ ./src/
COPY .env.production .env

# Create necessary directories and set permissions
RUN mkdir -p /app/logs && \
    chown -R orcastrate:orcastrate /app

# Switch to non-root user
USER orcastrate

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV PYTHONPATH="/app"
ENV ENVIRONMENT="production"

# Default command
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
