FROM python:3

COPY . /app

WORKDIR ./app

RUN chmod +x ./setup.sh
RUN ./setup.sh

