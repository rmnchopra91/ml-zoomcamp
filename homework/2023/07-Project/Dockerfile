FROM python:3.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

RUN mkdir -p /artifacts

COPY ["artifacts/*", "./artifacts/"]

COPY ["preprocess.py", "load_model.py", "web_api.py", "./"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "web_api:app"]