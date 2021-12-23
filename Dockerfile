FROM python:3.10

COPY requirements.txt /docker/requirements.txt
WORKDIR /docker
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . /docker