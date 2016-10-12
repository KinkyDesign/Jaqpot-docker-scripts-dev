cd packages/jaqpot-api
git pull
mvn clean install -P JaqpotQuattroTest -D skipTests=true
cd ../../ 
docker build -t jaqpot/api .
docker stop jaqpot-api
docker rm jaqpot-api
docker run -it -d -p 8081:8080 --restart=unless-stopped --name jaqpot-api jaqpot/api
