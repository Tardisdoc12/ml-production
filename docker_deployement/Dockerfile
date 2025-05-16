FROM python:3.8-slim
WORKDIR /app
COPY . .
COPY ../data/models/ ./data/models/
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "scripts.serve_model:app", "--host", "0.0.0.0", "--port", "8000"]