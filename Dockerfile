FROM continuumio/miniconda3

RUN mkdir -p jh2284

COPY . /jh2284

WORKDIR /jh2284

RUN conda env update --file environment.yml --name base
