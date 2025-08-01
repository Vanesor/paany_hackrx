# Dockerfile using best practices

# --- Stage 1: The Builder ---
# Use a specific version for reproducibility
FROM python:3.11.9-slim-bookworm as builder

# Set up the virtual environment
WORKDIR /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies
# Using --no-cache-dir reduces layer size
COPY requirements.txt .
RUN python -m venv . && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Stage 2: The Runner ---
FROM python:3.11.9-slim-bookworm

WORKDIR /app

# Copy the virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv

# Create a non-root user to run the application
RUN useradd --create-home appuser
USER appuser

# Copy only the necessary application files, not the whole directory
COPY --chown=appuser:appuser . .

# Activate the virtual environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set default port (can be overridden by environment variable)
ENV PORT=10000

# Expose the port and define the healthcheck
EXPOSE $PORT
HEALTHCHECK CMD curl --fail http://localhost:$PORT/api/health || exit 1
CMD ["python", "main.py"]