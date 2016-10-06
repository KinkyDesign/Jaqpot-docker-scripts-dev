docker stop xvfb
docker rm xvfb
docker run -it -d --restart=unless-stopped --name xvfb jaqpot/xvfb
