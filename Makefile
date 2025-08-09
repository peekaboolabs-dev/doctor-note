.PHONY: help build up down logs shell test clean

help: ## 도움말 표시
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Docker 이미지 빌드
	docker-compose build

build-dev: ## 개발용 Docker 이미지 빌드
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml build

up: ## 컨테이너 시작
	docker-compose up -d

up-dev: ## 개발 모드로 컨테이너 시작
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

down: ## 컨테이너 중지 및 제거
	docker-compose down

logs: ## 컨테이너 로그 확인
	docker-compose logs -f medical-rag

shell: ## 컨테이너 쉘 접속
	docker-compose exec medical-rag /bin/bash

test: ## 테스트 실행
	docker-compose exec medical-rag pytest tests/

clean: ## 도커 볼륨 및 캐시 정리
	docker-compose down -v
	docker system prune -f

setup-local: ## 로컬 개발 환경 설정
	python -m venv venv
	source venv/bin/activate && pip install -r requirements.txt
	cp .env.example .env

run-local: ## 로컬에서 실행
	source venv/bin/activate && python main.py --mode search --query "고혈압"

init-db: ## 의학 데이터베이스 초기화
	docker-compose run --rm medical-rag python main.py --mode setup