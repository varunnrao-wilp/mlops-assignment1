FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app .

EXPOSE 5000

CMD ["flask", "--app", "main.py", "run", "--host=0.0.0.0", "--port=5000"]