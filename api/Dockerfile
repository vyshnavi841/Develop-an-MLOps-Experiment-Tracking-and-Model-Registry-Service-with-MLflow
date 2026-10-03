# Multi-stage Dockerfile for FastAPI MLOps Service

# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies in virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt


# Stage 2: Final Production Runtime Image
FROM python:3.10-slim AS runner

WORKDIR /app

# Install runtime system libraries (libpq for postgres client)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Create non-root application user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# Copy application source code
COPY config/ /app/config/
COPY api/ /app/api/
COPY ml_core/ /app/ml_core/
COPY tests/ /app/tests/

# Create directory for local mlruns artifact access
RUN mkdir -p /mlruns && chown -R appuser:appuser /app /mlruns

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
