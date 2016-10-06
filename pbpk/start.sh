#cd packages/
#git pull
#cd ../../ 
docker build -t pbpk .
docker stop pbpk
docker rm pbpk
docker run -it -d -p 8088:8088 --restart=unless-stopped --cpu-shares=256 --cpuset="0,1" --name pbpk pbpk
