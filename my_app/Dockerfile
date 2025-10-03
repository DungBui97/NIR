# Multi-stage build for optimization
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY my_app/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.10-slim AS production

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy application files
COPY my_app/app.py .
COPY my_app/config.py .
COPY my_app/models.py .
COPY my_app/indices ./indices

# Copy models directory
COPY mo_hinh ./mo_hinh

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Add local Python packages to PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Set default port
ENV PORT=8080

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Run the application
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1"]
