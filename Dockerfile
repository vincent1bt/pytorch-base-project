FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY /model /model
COPY /utils /utils
COPY /data /data

COPY hparameters.py .
COPY train.py .

ENTRYPOINT [ "train.py" ]