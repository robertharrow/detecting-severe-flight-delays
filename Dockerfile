FROM python:3.8

RUN apt-get update
RUN apt-get install nano

RUN mkdir -p /home/app
COPY ./app /home/app
WORKDIR /home/app
RUN pip3 install -r requirements.txt

CMD [ "gunicorn", "--timeout=120","--workers=5", "--threads=1", "-b 0.0.0.0:80", "app:server"]
