FROM python:2
MAINTAINER Charalampos Chomenidis <hampos@me.com>

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
RUN pip install sortedcontainers
#RUN pip install pattern

# Expose the ports we're interested in
EXPOSE 8000

ADD ./packages/UI /UI

RUN python /UI/manage.py migrate

# Set the default command to run on boot
# This will boot WildFly in the standalone mode and bind to all interface
CMD ["python","/UI/manage.py","runserver","0.0.0.0:8000"]
