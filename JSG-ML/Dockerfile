FROM python:3.9-slim


WORKDIR /app
RUN apt-get update && apt-get install -y build-essential coinor-cbc

COPY . /app

RUN pip install --upgrade pip setuptools
RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
