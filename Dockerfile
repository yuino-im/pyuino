FROM python:3.13-slim AS builder
WORKDIR /opt/yuino/

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY requirements-srv.txt ./requirements.txt
COPY .git ./.git

RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir -r requirements.txt

ADD pyuino ./pyuino
RUN pip install .

# runner
FROM python:3.13-slim

WORKDIR /opt/pyuino/

COPY --from=builder /usr/local/lib/python3.13/site-packages/ /usr/local/lib/python3.13/site-packages/
COPY --from=builder /usr/local/bin/pyuino /usr/local/bin/pyuino

ENTRYPOINT ["pyuino"]
