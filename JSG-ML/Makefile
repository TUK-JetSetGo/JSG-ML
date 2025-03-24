IMAGE_NAME=pulp-app
PORT=8000
CONTAINER_NAME=pulp-app-container

default: build run

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run --rm --name $(CONTAINER_NAME) -p $(PORT):$(PORT) $(IMAGE_NAME)

run-ci:
	docker run -d --rm --name $(CONTAINER_NAME) -p $(PORT):$(PORT) $(IMAGE_NAME)

stop:
	-docker stop $(CONTAINER_NAME)

clean:
	docker rmi $(IMAGE_NAME)