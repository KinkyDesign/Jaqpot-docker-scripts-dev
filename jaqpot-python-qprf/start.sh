docker stop python-qprf
docker rm python-qprf
cd packages/qprf
git pull
cd ../../ 
docker build -t python-qprf .
docker run -it -d -p 8094:5000 --restart=unless-stopped --name python-qprf  python-qprf
