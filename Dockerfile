FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN mkdir -p /opennre
WORKDIR /opennre

COPY . /opennre
COPY ./07301f59bb8cb113803be316267f06ddf9243cdbba92a4c8067ef92442d2c574.554244d3476d97501a766a98078421817b14654496b86f2f7bd139dc502a4f29 /root/.flair/models/ner-english-large/
COPY ./wiki80_bert_softmax.pth.tar /root/.opennre/pretrain/nre/
COPY ./pretrain/bert-base-uncased /root/.opennre/pretrain/

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install jupyter
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade transformers
RUN pip install --no-cache-dir fastapi==0.63.0 uvicorn==0.13.4

CMD ["/bin/bash"]
