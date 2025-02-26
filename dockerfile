# Step 1: Use a lightweight Python base image
FROM python:3.9-slim

# Step 2: Install system dependencies for OpenCV, PyTorch, and other libraries
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libopenblas0 \
    libgomp1 \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Set the working directory in the container
WORKDIR /app

# Step 4: Copy the requirements file into the container
COPY requirements.txt .

# Step 5: Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of the application code
COPY . .

# Step 7: Expose the port the app runs on
EXPOSE 5000

# Step 8: Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Step 9: Run the Flask app with Gunicorn for better performance
CMD ["python", "app.py"]