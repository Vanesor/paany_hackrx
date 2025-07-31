# Dockerfile using requirements.txt

# --- Stage 1: The Builder ---
FROM python:3.11 as builder

WORKDIR /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .
RUN python -m venv . && \
    pip install --no-cache-dir -r requirements.txt

# --- Stage 2: The Runner ---
FROM python:3.11-slim

WORKDIR /app

# Copy the virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv

# Copy your application code
COPY . .

# Activate the virtual environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Expose the port and run the application
EXPOSE 10000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]