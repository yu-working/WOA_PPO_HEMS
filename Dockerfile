FROM python:3.10.18-slim

WORKDIR /AI_decision_pmv_balance

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 複製整個專案內容到容器內
COPY . /AI_decision_pmv_balance

# 安裝 Python 相依套件
RUN pip install --no-cache-dir -r requirements.txt

# 開放 5000 port（供 API 使用）
EXPOSE 5000

# 啟動主程式
CMD ["python", "pmv_balance_api.py"]