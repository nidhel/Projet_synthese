FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY scripts/api.py ./app.py
COPY models/model.pkl ./models/model.pkl

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
