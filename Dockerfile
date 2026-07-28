FROM python:3.14-slim

LABEL maintainer="TheChef <thechef3179>"

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

# Copy application files
COPY main.py README.md requirements.txt ./
RUN mkdir logs
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Install NiceGUI from source
RUN pip install -r requirements.txt

ARG DEFAULT_PORT=8000
ENV APP_PORT=${DEFAULT_PORT}
EXPOSE ${APP_PORT}

ENTRYPOINT ["/entrypoint.sh"]
