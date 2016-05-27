docker stop python-readacross
docker rm python-readacross
cd packages/readacross
git pull
cd ../../ 
docker build -t python-readacross .
docker run -it -d -p 8093:5000 --restart=unless-stopped --name python-readacross  python-readacross
