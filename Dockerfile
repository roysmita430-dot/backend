# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements file first
COPY requirements.txt .

# Upgrade pip, setuptools, wheel first
RUN pip install --upgrade pip setuptools wheel

# Install all dependencies
RUN pip install -r requirements.txt

# Copy all project files
COPY . .

# Expose port if needed (adjust according to your app)
EXPOSE 8000

# Command to run your app (adjust if you use Flask, FastAPI, etc.)
CMD ["python", "backend.py"]


