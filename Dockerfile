# ==========================================
# Base Stage
# ==========================================
FROM python:3.13-slim AS base

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

# Copy only dependency files first
COPY pyproject.toml uv.lock* README.md ./

# Install dependencies into the System Python (/usr/local)
RUN uv pip install --system --no-cache .

# ==========================================
# Runtime Stage
# ==========================================
FROM base AS runtime

# Copy the pre-installed libraries from the builder stage
COPY --from=builder /usr/local /usr/local

# Copy your application source code
COPY mylib ./mylib
COPY templates ./templates
COPY main.py .

# Expose the port
EXPOSE 8000

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]