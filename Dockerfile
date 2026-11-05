FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies for SQLite, PyTorch, and HMM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files and install Python packages
COPY pyproject.toml requirements.txt* ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    pandas numpy pyarrow yfinance scikit-learn lightgbm torch \
    optuna mlflow shap hmmlearn statsmodels pydantic pyyaml \
    python-dotenv apscheduler fastapi uvicorn httpx streamlit plotly \
    sentence-transformers chromadb rank-bm25 feedparser pytest

# Copy application source code and configurations
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY prompts/ /app/prompts/
COPY data/ /app/data/
COPY pytest.ini /app/

ENV PYTHONPATH=/app/src
ENV TZ=Asia/Kolkata
ENV MODE=PAPER

EXPOSE 8000 8501 5000

CMD ["uvicorn", "stock_engine.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

