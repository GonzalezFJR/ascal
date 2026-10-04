# ascal web app (public calibration service). Build from the repository root:
#   docker build -f deploy/Dockerfile -t ascal-web .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
    ASCAL_WEB_DATA=/data ASCAL_WEB_EXAMPLES=/app/examples/web
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY ascal ./ascal
RUN pip install --no-cache-dir ".[web,raw]"
COPY examples ./examples
RUN useradd -u 10001 -m ascal && mkdir -p /data && chown ascal /data
USER ascal
EXPOSE 8000
CMD ["uvicorn", "ascal.web.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
