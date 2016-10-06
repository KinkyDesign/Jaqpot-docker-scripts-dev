FROM jaqpot/wildfly10
MAINTAINER Charalampos Chomenidis <hampos@me.com>

ENV DISPLAY xvfb:0.0

#ADD ./scripts/standalone.xml /opt/jboss/wildfly/standalone/configuration/standalone.xml
ADD ./packages/imageAnalysis/target/imageAnalysis.war /opt/jboss/wildfly/standalone/deployments/imageAnalysis.war

