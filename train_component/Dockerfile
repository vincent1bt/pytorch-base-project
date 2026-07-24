FROM python:3.11

WORKDIR /app
COPY main.py .

ARG GIT_SHA
ENV GIT_SHA=${GIT_SHA}
LABEL git_sha="${GIT_SHA}"

ENTRYPOINT [ "main.py" ]