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

# Expose the port and define the healthcheck
EXPOSE 4000
HEALTHCHECK CMD curl --fail http://localhost:4000/api/health || exit 1
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4000"]