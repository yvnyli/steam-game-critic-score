FROM python:3.11-slim

# Copy the 'uv' and 'uvx' executables from the latest uv image into /bin/ in this image
# 'uv' is a fast Python package installer and environment manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory inside the container to /app
# All subsequent commands will be run from here
WORKDIR /app

# Add the virtual environment's bin directory to the PATH so Python tools work globally
ENV PATH="/app/.venv/bin:$PATH"

# Copy the project configuration files into the container
# pyproject.toml     → project metadata and dependencies
# uv.lock            → locked dependency versions (for reproducibility)
# .python-version    → Python version specification
COPY "pyproject.toml" "uv.lock" ".python-version" ./

# Copy required project files into the container
COPY app ./app
COPY interactive ./interactive
COPY models ./models
COPY data ./data

# Install dependencies exactly as locked in uv.lock, without updating them
RUN uv sync --locked

# Expose TCP port 8080 so it can be accessed from outside the container
EXPOSE 8080

# Run the application using uvicorn (ASGI server)
# predict:app → refers to 'app' object inside predict.py
# --host 0.0.0.0 → listen on all interfaces
# --port 8080    → listen on port 8080
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
#--------------------------------------------------------------------