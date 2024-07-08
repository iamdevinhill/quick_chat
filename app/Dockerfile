# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy all files from the current directory to the container
COPY . .

# Install FastAPI and required dependencies
RUN pip install --no-cache-dir fastapi uvicorn transformers

# Expose port 8000 to the outside world
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
