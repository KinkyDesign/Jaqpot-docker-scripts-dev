docker stop python-validation
docker rm python-validation
cd packages/validation
git pull
cd ../../ 
docker build -t python-validation .
docker run -it -d -p 8092:5000 --restart=unless-stopped --name python-validation  python-validation
