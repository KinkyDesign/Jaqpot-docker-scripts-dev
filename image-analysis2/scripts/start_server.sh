#!/bin/sh
Xvfb :0 -screen 0 800x600x16 &
/opt/jboss/wildfly/bin/standalone.sh --server-config=standalone.xml 
