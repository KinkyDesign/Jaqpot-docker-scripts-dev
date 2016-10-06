cd packages/readacross
git pull
cd ../../ 
docker build -t python-readacross2 .
docker stop python-readacross2
docker rm python-readacross2
docker run -it -d -p 8095:5000 --restart=unless-stopped --name python-readacross2  python-readacross2
