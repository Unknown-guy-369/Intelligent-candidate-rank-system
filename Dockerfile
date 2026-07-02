FROM python:3.11-slim

WORKDIR /code
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

COPY requirements.txt requirements-sandbox.txt ./
RUN pip install --no-cache-dir -r requirements-sandbox.txt

COPY app ./app
COPY sandbox_app.py ./sandbox_app.py

EXPOSE 7860
CMD ["python", "sandbox_app.py"]
