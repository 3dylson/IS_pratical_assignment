ARG UBUNTU_VERSION=20.04
FROM ubuntu:$UBUNTU_VERSION

RUN apt update --fix-missing && \
    apt install -y --no-install-recommends ca-certificates git sudo curl libgl1-mesa-glx python3-pip libglib2.0-0 && \
    apt clean

RUN mkdir /app

COPY /requirements.txt /app
COPY /datasets /app
COPY /FashionCNN.py /app
COPY /FashionDataset.py /app
COPY /main.py /app
COPY /Model.py /app
COPY /trainedModel.pth /app

RUN pip3 install --upgrade pip
RUN pip3 install -r /app/requirements.txt

WORKDIR /app

CMD [ "python3", "./main.py" ]