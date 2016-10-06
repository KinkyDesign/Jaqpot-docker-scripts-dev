cd packages/imageAnalysis
git pull
mvn clean install  -D skipTests=true
cd ../../ 
docker build -t jaqpot/image-analysis .
docker stop image-analysis
docker rm image-analysis
docker run -it -d -p 8880:8080 --restart=unless-stopped --link xvfb --name image-analysis jaqpot/image-analysis
