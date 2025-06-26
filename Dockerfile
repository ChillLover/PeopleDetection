FROM python:3.12-slim

WORKDIR /pd

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY best.pt .

COPY all.py .

CMD ["python", "all.py"]
