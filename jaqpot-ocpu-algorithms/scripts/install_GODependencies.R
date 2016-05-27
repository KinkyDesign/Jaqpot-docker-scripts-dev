#!/usr/bin/env Rscript
install.packages("RCurl", repos="http://cran.cc.uoc.gr/mirrors/CRAN/")
require(devtools)
install_version("jsonlite", version = "0.9.16", repos = "http://cran.cc.uoc.gr/mirrors/CRAN/")

install.packages("pmml", repos="http://cran.cc.uoc.gr/mirrors/CRAN/")
install.packages("blockcluster", repos="http://cran.cc.uoc.gr/mirrors/CRAN/")
install.packages("Matrix", repos="http://cran.cc.uoc.gr/mirrors/CRAN/")
install.packages("vegan", repos="http://cran.cc.uoc.gr/mirrors/CRAN/")

#install.packages("jsonlite", repos="http://cran.cc.uoc.gr/mirrors/CRAN/")

source("http://bioconductor.org/biocLite.R")
biocLite("org.Hs.eg.db")
biocLite("GSEABase")
biocLite("GOstats")
biocLite("Rgraphviz")
