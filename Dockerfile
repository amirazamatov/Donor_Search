FROM python:3.10.0

COPY requirements.txt /app/requirements.txt
COPY src /app

WORKDIR /app

RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


CMD ["python", "app.py"]