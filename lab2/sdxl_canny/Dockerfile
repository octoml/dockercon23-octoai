FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG SERVING_PORT=8080
ENV SERVING_PORT=$SERVING_PORT

WORKDIR /app

# Install necessary apt packages
RUN apt update && \
    apt install -y python3-pip && \
    apt install -y libgl1-mesa-glx libglib2.0-0

# Upgrade pip and install the copied in requirements.
RUN pip install --no-cache-dir --upgrade pip
ADD requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Include a download.py file that can pre-load the model files.
ADD download.py .

# Fetch the model and cache it locally.
RUN python3 download.py

# Copy in the files necessary to fetch, run and serve the model.
ADD model.py .
ADD server.py .

# Expose the serving port.
EXPOSE $SERVING_PORT

# Run the server to handle inference requests.
CMD python3 -u server.py
