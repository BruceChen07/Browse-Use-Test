# Use an official Python runtime as a parent image
FROM python:3.11-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for Playwright and others
# We install curl to download uv
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy the project files into the container
COPY pyproject.toml uv.lock ./
COPY main.py ./
COPY README.md ./

# Install dependencies using uv
# --frozen ensures we use the exact versions in uv.lock
RUN uv sync --frozen

# Install Playwright browsers
# We need to use the virtual environment's python to run playwright
RUN uv run playwright install --with-deps

# Copy the rest of the application code
# (In this case, we already copied main.py, but for larger projects this is useful)
# COPY . .

# Set environment variables
# Python won't buffer output, so we see logs immediately
ENV PYTHONUNBUFFERED=1
# Add the virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Run the application
CMD ["python", "main.py"]
