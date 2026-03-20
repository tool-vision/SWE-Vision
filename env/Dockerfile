FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    ipykernel \
    matplotlib \
    numpy \
    opencv-python-headless \
    pandas \
    Pillow \
    plotly \
    scikit-image \
    scipy \
    seaborn

WORKDIR /mnt/data

CMD ["sleep", "infinity"]
