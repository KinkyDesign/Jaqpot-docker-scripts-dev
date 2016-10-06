docker stop xvfb
docker rm xvfb
docker run -d -e DISPLAY=0 --restart=unless-stopped --name xvfb jaqpot/xvfb
