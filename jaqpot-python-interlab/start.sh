docker stop python-interlab
docker rm python-interlab
cd packages/PWS
git pull
cd ../../ 
docker build -t python-interlab .
docker run -it -d -p 8091:5000 --restart=unless-stopped --name python-interlab  python-interlab
