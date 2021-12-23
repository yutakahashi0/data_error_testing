sudo docker build -t py310 .

sudo docker run --rm \
-it \
--name py310 \
--mount type=bind,source="$(pwd)",target=/usr/docker \
--workdir /usr/docker \
python:3.10 /bin/bash