FROM tensorflow/tensorflow:1.14.0-py3

WORKDIR /home

ENV AT16K_RESOURCES_DIR=/home/.at16k

RUN apt update \
    && pip install at16k \
    && python -m at16k.download all

ENTRYPOINT [ "at16k-serve" ]