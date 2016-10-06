FROM jaqpot/wildfly10
MAINTAINER Charalampos Chomenidis <hampos@me.com>

RUN apk --update add ttf-dejavu

ENV DISPLAY xvfb:0

#ADD ./scripts/standalone.xml /opt/jboss/wildfly/standalone/configuration/standalone.xml
ADD ./packages/imageAnalysis/target/imageAnalysis.war /opt/jboss/wildfly/standalone/deployments/imageAnalysis.war

CMD ["/opt/jboss/wildfly/bin/standalone.sh", "-b", "0.0.0.0", "-Djava.awt.headless=false"]
