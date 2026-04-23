FROM apache/airflow:2.8.0-python3.11
RUN pip install --no-cache-dir \
    yfinance>=1.2.0 \
    pandas>=2.0 \
    numpy>=1.24 \
    scipy>=1.11 \
    psycopg2-binary>=2.9
