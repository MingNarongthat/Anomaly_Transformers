up:
	docker-compose up -d

down:
	docker-compose down -v

pull:
	docker-compose pull

logs:
	docker-compose logs -f

build:
	docker-compose build

run:
	docker exec -it transformermodel-transformergpuv2-1 bash