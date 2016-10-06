FROM alpine
MAINTAINER Charalampos Chomenidis <hampos@me.com>

RUN set -xe; \
    apk add --update xvfb;\
    rm -rf /var/cache/apk/*;

CMD ["Xvfb",":0","-screen","0","800x600x16"]
