FROM tensorflow/tensorflow:latest

WORKDIR /opt/bp-jupyter

ADD ./ /opt/bp-jupyter/

LABEL version='0.1.0'
LABEL maintainer="Giang Nguyen <giang.nguyen@savba.sk>"
LABEL description="Data Science stack"

ENV LANG C.UTF-8

RUN apt-get update && \
    pip3 install --upgrade pip

COPY requirements.txt /code/

RUN pip install -r requirements.txt

RUN rm requirements.txt

RUN chmod +x run.sh

EXPOSE 8888
CMD ["./run.sh"]