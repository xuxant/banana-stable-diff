FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

RUN apt-get update && apt-get install -y git

RUN pip3 install --upgrade pip

ADD requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

ENV HUGGING_FACE_HUB_TOKEN=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

ADD app.py .

ADD download.py .

RUN python3 download.py

EXPOSE 8000


CMD python3 app.py