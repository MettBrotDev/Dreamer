FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Install dependencies including Python, pip, and swig
RUN apt-get update \
    && apt-get install -y git build-essential python3 python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Install additional pip requirements
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden
CMD ["python3", "NetTraining.py"]