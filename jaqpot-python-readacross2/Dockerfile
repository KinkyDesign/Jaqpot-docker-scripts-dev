FROM python:2
MAINTAINER Charalampos Chomenidis <hampos@me.com>

RUN pip install --upgrade pip
RUN pip install flask
RUN pip install numpy
RUN pip install scikit-learn
RUN apt-get update
RUN apt-get install -y libatlas-base-dev gfortran libopenblas-dev
RUN pip install scipy
RUN pip install matplotlib

# Expose the ports we're interested in
EXPOSE 5000

ADD ./packages/readacross/rac.py /rac.py

# Set the default command to run on boot
# This will boot WildFly in the standalone mode and bind to all interface
CMD ["python","/rac.py"]

