FROM python:3.9

ADD lstm.h5 .
ADD vision.py . 
ADD playground.py . 

RUN pip install numpy 
RUN pip install opencv-python
RUN pip install tensorflow
RUN pip install scikit-learn
RUN pip install winsound
RUN pip install keras
RUN pip install time
RUN pip install mediapipe 



CMD ['python', './playground.py']