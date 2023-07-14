FROM python:3.10.6-bullseye

RUN apt-get update && apt-get install libgl1 -y
RUN apt-get install libopenblas-dev -y

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit",  "run", "main.py",  "--browser.serverAddress", "0.0.0.0", "--server.port", "8000"]
