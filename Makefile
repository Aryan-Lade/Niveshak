.PHONY: install test lint format reproduce api dashboard docker-up docker-down

install:
	pip install --upgrade pip
	pip install -e .

test:
	python -m pytest tests/ -v

lint:
	ruff check .

format:
	ruff format .

reproduce:
	python scripts/reproduce.py

api:
	uvicorn stock_engine.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run src/stock_engine/dashboard/app.py --server.port 8501

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

