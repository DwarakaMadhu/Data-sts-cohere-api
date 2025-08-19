
# Simple container for FastAPI STS app
FROM python:3.11-slim

# System deps (optional, for faster builds keep minimal)
RUN pip install --no-cache-dir --upgrade pip

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Expose port and run
EXPOSE 10000
ENV PORT=10000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "10000"]
