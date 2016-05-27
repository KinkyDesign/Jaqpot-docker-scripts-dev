cd packages/Jaqpot
git pull
mvn clean install -P JaqpotQuattroTest -D skipTests=true
cd ../../ 
docker build -t jaqpot-java-core .
docker stop jaqpot
docker rm jaqpot
docker run -it -d -p 8080:8080 -p 8787:8787 -h test.jaqpot --volumes-from javamelody --restart=unless-stopped --name jaqpot jaqpot-java-core
