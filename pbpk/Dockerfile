FROM ubuntu:14.04
MAINTAINER Charalampos Chomenidis <hampos@me.com>

RUN apt-get update
RUN apt-get install -y python-software-properties
RUN apt-get install -y software-properties-common
RUN apt-add-repository ppa:fkrull/deadsnakes-python2.7
RUN apt-get update
RUN apt-get install -y python2.7 python2.7-dev
RUN apt-get -y install python-pip
RUN pip install --upgrade pip
RUN pip install django==1.8.3
RUN pip install django-endless-pagination
RUN pip install django-simple-captcha
RUN pip install django-ratings
RUN pip install jsonpatch
RUN pip install requests
RUN pip install xmltodict
RUN pip install django-haystack
RUN pip install elasticsearch
RUN pip install pathlib
RUN pip install wget
RUN pip install importlib
RUN pip install django-allauth
RUN pip install numpy
RUN apt-get install -y libfreetype6-dev libxft-dev
RUN pip install --upgrade setuptools
RUN pip install matplotlib
RUN pip install control
#RUN pip install pattern
RUN apt-get build-dep -y python-scipy
RUN pip install slycot
RUN pip install cvxopt

# Expose the ports we're interested in
EXPOSE 8000

ADD ./packages/pbpk /pbpk

#RUN python /UI/manage.py migrate

# Set the default command to run on boot
# This will boot WildFly in the standalone mode and bind to all interface
CMD ["python","/pbpk/manage.py","runserver","0.0.0.0:8088"]
