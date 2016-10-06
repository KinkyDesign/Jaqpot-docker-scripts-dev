FROM jboss/base-jdk:8
MAINTAINER Charalampos Chomenidis <hampos@me.com>

# Set the WILDFLY_VERSION env variable
ENV WILDFLY_VERSION 8.2.0.Final

# Add the WildFly distribution to /opt, and make wildfly the owner of the extracted tar content
# Make sure the distribution is available from a well-known place
RUN cd $HOME && curl http://download.jboss.org/wildfly/$WILDFLY_VERSION/wildfly-$WILDFLY_VERSION.tar.gz | tar zx && mv $HOME/wildfly-$WILDFLY_VERSION $HOME/wildfly

USER root

#ADD ./scripts/fastestmirror.conf /etc/yum/pluginconf.d/fastestmirror.conf

#RUN yum clean all && yum swap fakesystemd systemd
RUN yum install -y xorg-x11-server-Xvfb
#RUN Xvfb :0 -screen 0 800x600x16 

# Set the JBOSS_HOME env variable
ENV JBOSS_HOME /opt/jboss/wildfly
ENV DISPLAY :0.0
#ENV JAVA_OPTS -Xdebug -Xrunjdwp:transport=dt_socket,address=8787,server=y,suspend=n

# Expose the ports we're interested in
EXPOSE 8080
#EXPOSE 8787

ADD ./scripts/standalone.xml /opt/jboss/wildfly/standalone/configuration/standalone.xml
ADD ./scripts/start_server.sh /opt/jboss/wildfly/bin/start_server.sh
#ADD ./packages/imageAnalysis.war /opt/jboss/wildfly/standalone/deployments/imageAnalysis.war
ADD ./packages/imageAnalysis/target/imageAnalysis.war /opt/jboss/wildfly/standalone/deployments/imageAnalysis.war

# Set the default command to run on boot
# This will boot WildFly in the standalone mode and bind to all interface
#CMD ["/opt/jboss/wildfly/bin/standalone.sh", "-b", "0.0.0.0"]
CMD ["/opt/jboss/wildfly/bin/start_server.sh"]
