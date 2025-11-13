FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the trained model
COPY . .

# Expose the port where FastAPI runs
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
# 'app:app' means look for the variable 'app' inside the file 'app.py'
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]