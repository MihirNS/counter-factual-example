FROM python:3.7.13

COPY . .

RUN pip install carla-recourse protobuf==3.20.1 torch==1.7.0+cpu torchvision==0.8.1+cpu imbalanced-learn -f https://download.pytorch.org/whl/torch_stable.html
