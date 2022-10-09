FROM python:3.8

RUN apt-get update
RUN apt-get install nano

RUN mkdir wd
WORKDIR wd
COPY data/prepared/ data/prepared/
COPY app/requirements.txt .
RUN pip3 install -r requirements.txt

COPY app/ ./

CMD [ "gunicorn", "--timeout=120","--workers=5", "--threads=1", "-b 0.0.0.0:80", "app:server"]
