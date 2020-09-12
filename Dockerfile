FROM python:3.8-slim-buster

RUN apt-get update -y

RUN apt-get install -y python-pip
RUN apt-get install -y libgl1-mesa-glx 

RUN pip install Cython
RUN pip install numpy>=1.18.5
RUN pip install opencv-python
RUN pip install torch>=1.5.1+cu101
RUN pip install matplotlib
RUN pip install pillow
RUN pip install tensorboard
RUN pip install PyYAML>=5.3
RUN pip install torchvision>=0.6.1+cu101
RUN pip install scipy
RUN pip install tqdm
RUN pip install flask
RUN pip install flask-cors
RUN pip install Flask gunicorn


COPY . .

ENV PORT 8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app