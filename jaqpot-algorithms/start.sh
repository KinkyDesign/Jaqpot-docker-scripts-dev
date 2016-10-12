cd packages/jaqpot-algorithms
git pull
mvn clean install  -D skipTests=true
cd ../../ 
docker build -t jaqpot/algorithms .
docker stop jaqpot-algorithms
docker rm jaqpot-algorithms
docker run -it -d -p 8090:8080 --restart=unless-stopped --name jaqpot-algorithms jaqpot/algorithms
