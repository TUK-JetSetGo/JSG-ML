IMAGE_NAME=pulp-app
PORT=8000

default: build run

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run --rm -p $(PORT):$(PORT) $(IMAGE_NAME)

clean:
	docker rmi $(IMAGE_NAME)
