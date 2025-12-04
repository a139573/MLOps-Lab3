# Use a single, clean stage
FROM python:3.13-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

# Install uv directly
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock* README.md ./

# Copy your ACTUAL source code
COPY mylib ./mylib
COPY main.py .
COPY templates .

# Install dependencies
RUN uv pip install --system --no-cache .

# Expose port
EXPOSE 8000

# Start command: Pointing to main.py instead of api/api.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]