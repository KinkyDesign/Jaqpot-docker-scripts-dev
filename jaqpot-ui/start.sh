cd packages/UI
git pull
cd ../../ 
docker build -t jaqpot-ui .
docker stop jaqpot-ui
docker rm jaqpot-ui
docker run -it -d -p 8000:8000 --restart=unless-stopped --cpu-shares=256 --cpuset="0,1" --name jaqpot-ui jaqpot-ui
