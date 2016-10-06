docker stop image-analysis
docker rm image-analysis
cd packages/imageAnalysis
git pull
mvn clean install  -D skipTests=true
cd ../../ 
docker build -t image-analysis .
docker run -it -d -p 8880:8880 --restart=unless-stopped --name image-analysis image-analysis
