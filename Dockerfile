FROM python:3.14-slim

LABEL maintainer="TheChef <thechef3179>"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

# Copy application files
COPY main.py README.md requirements.txt ./
RUN mkdir logs

# Install NiceGUI from source
RUN uv pip install -r requirements.txt

ARG DEFAULT_PORT=8000
ENV APP_PORT=${DEFAULT_PORT}
EXPOSE ${APP_PORT}

CMD ["/opt/venv/bin/python", "main.py"]
