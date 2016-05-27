cd packages/algorithms
git pull
cd ../../ 
docker build -t python-algorithms .
docker stop python-algorithms
docker rm python-algorithms
docker run -it -d -p 8089:5000 --restart=unless-stopped --name python-algorithms  python-algorithms
