FROM jaqpot/wildfly10
MAINTAINER Charalampos Chomenidis <hampos@me.com>

ADD ./packages/jaqpot-algorithms/target/jaqpot-algorithms-4.0.2.war /opt/jboss/wildfly/standalone/deployments/jaqpot-algorithms-4.0.2.war

CMD ["/opt/jboss/wildfly/bin/standalone.sh", "-b", "0.0.0.0"]
