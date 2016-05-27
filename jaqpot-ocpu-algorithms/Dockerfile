FROM opencpu/base
MAINTAINER Charalampos Chomenidis <hampos@me.com>
RUN apt-get -y install libxml2-dev libcurl4-openssl-dev

COPY ./scripts/install_devtools.R /scripts/ 
RUN R CMD BATCH /scripts/install_devtools.R /dev/stdout

COPY ./scripts/install_GODependencies.R /scripts/
RUN R CMD BATCH /scripts/install_GODependencies.R /dev/stdout 
COPY ./scripts/server.conf /etc/opencpu/server.conf

COPY ./packages/GOdescrCalculus_1.0.tar.gz /packages/
#RUN R CMD check /packages/GOdescrCalculus_1.0.tar.gz
RUN R CMD INSTALL /packages/GOdescrCalculus_1.0.tar.gz --library=/usr/local/lib/R/site-library

COPY ./packages/PLMplusPMMLpkg_1.0.tar.gz /packages/
RUN R CMD INSTALL /packages/PLMplusPMMLpkg_1.0.tar.gz --library=/usr/local/lib/R/site-library

COPY ./scripts/install_GLMNETDependencies.R /scripts/
RUN R CMD BATCH /scripts/install_GLMNETDependencies.R /dev/stdout
COPY ./packages/glmNETpkg_1.0.tar.gz /packages/
RUN R CMD INSTALL /packages/glmNETpkg_1.0.tar.gz --library=/usr/local/lib/R/site-library

COPY ./scripts/install_ClusteringDependencies.R /scripts/
RUN R CMD BATCH /scripts/install_ClusteringDependencies.R /dev/stdout
COPY ./packages/clusteringPkg_1.0.tar.gz /packages/
RUN R CMD INSTALL /packages/clusteringPkg_1.0.tar.gz --library=/usr/local/lib/R/site-library

COPY ./scripts/install_ExpDesignDependencies.R /scripts/
RUN R CMD BATCH /scripts/install_ExpDesignDependencies.R /dev/stdout
COPY ./packages/ExpDesignPkg_1.5.tar.gz /packages/
RUN R CMD INSTALL /packages/ExpDesignPkg_1.5.tar.gz --library=/usr/local/lib/R/site-library

CMD service opencpu restart && tail -F /var/log/opencpu/apache_access.log
