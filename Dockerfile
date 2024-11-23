# Use an official Python image as the base
FROM python:3.10-slim

# Set a working directory inside the container
WORKDIR /app

# Copy the requirements directly into the container
COPY requirements.txt .

# Install necessary system dependencies and Python dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    portaudio19-dev \
    && pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Set a default command to keep the container running
CMD ["bash"]

# docker build -t mte252-audio-project .
# docker run -it --rm --name audio-container -v "$(pwd)":/app -w /app mte252-audio-project