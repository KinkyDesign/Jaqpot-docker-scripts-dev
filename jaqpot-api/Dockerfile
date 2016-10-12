FROM jaqpot/wildfly10
MAINTAINER Charalampos Chomenidis <hampos@me.com>

ADD ./scripts/standalone.xml /opt/jboss/wildfly/standalone/configuration/standalone.xml

ADD ./packages/jaqpot-api/target/jaqpot-api-4.0.3.war /opt/jboss/wildfly/standalone/deployments/jaqpot-api-4.0.3.war

CMD ["/opt/jboss/wildfly/bin/standalone.sh", "-b", "0.0.0.0"]
