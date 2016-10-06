FROM jaqpot/java
MAINTAINER Charalampos Chomenidis <hampos@me.com>

RUN apk --update add curl tar

RUN mkdir -p /opt/jboss

WORKDIR /opt/jboss

ENV WILDFLY_VERSION 10.0.0.Final
ENV WILDFLY_SHA1 c0dd7552c5207b0d116a9c25eb94d10b4f375549

ENV DOWNLOAD_URL=http://download.jboss.org/wildfly/$WILDFLY_VERSION

RUN curl -O https://download.jboss.org/wildfly/$WILDFLY_VERSION/wildfly-$WILDFLY_VERSION.tar.gz \
       && sha1sum wildfly-$WILDFLY_VERSION.tar.gz | grep $WILDFLY_SHA1 \
       && tar xz -f wildfly-$WILDFLY_VERSION.tar.gz \
       && rm wildfly-$WILDFLY_VERSION.tar.gz

RUN ln -s wildfly-$WILDFLY_VERSION wildfly

EXPOSE 8080 9990

ENV JBOSS_HOME /opt/jboss/wildfly

#RUN /opt/jboss/wildfly/bin/add-user.sh admin pass --silent

CMD ["/opt/jboss/wildfly/bin/standalone.sh", "-b", "0.0.0.0", "-bmanagement", "0.0.0.0"]
