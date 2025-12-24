FROM python:3.10-slim

WORKDIR /app

# Copy all project files
COPY . .

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

RUN mkdir -p ~/.streamlit
RUN bash -c 'echo -e "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = 8501\n\
" > ~/.streamlit/config.toml'

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
