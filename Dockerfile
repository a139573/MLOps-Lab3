# ==========================================
# Base Stage
# ==========================================
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and set buffer settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

# ==========================================
# Builder Stage
# ==========================================
FROM base AS builder

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock* README.md ./

# Copy mylib
COPY mylib ./mylib

# Install dependencies into the System Python (/usr/local)
# First pin the cpu versions of torch
RUN uv pip install --system --no-cache torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Download the rest
RUN uv pip install --system --no-cache .

# ==========================================
# Runtime Stage
# ==========================================
FROM base AS runtime

# Copy the pre-installed libraries from the builder stage
COPY --from=builder /usr/local /usr/local

# Copy application files
COPY mylib ./mylib
COPY cli ./cli
COPY api ./api
COPY templates ./templates

# Copy the serialized model and labels
COPY production_models ./production_models

# Expose the port
EXPOSE 8000

# Start command
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]