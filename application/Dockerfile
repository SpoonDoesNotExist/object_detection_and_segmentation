FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app
RUN python -m pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

COPY .. /app

RUN mkdir -p /app/application/data/upload_files /app/application/data/logs, /app/application/data/metrics

RUN date >> /app/application/build_date.txt

EXPOSE 5000
CMD ["python3", "main.py"]