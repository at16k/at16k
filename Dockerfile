FROM tensorflow/tensorflow:1.14.0-py3

RUN apt update \
    && pip install at16k \
    && python -m at16k.download all

ENTRYPOINT [ "at16k-convert" ]