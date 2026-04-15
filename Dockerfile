FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenTURNS, scipy, numpy, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran libopenblas-dev cmake swig && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies — cached layer (only rebuilds when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Ensure output directory exists
RUN mkdir -p figures

# Volume for retrieving generated figures on your host machine
VOLUME ["/app/figures"]

# ENTRYPOINT so you can append flags like --quick
ENTRYPOINT ["python", "main.py"]

# ─────────────────────────────────────────────────────────
# USAGE:
#
#   Build the image:
#     docker build -t bayesian-uq .
#
#   Run the FULL pipeline (~15-20 min, needs ~4 GB RAM):
#     docker run --rm -v $(pwd)/figures:/app/figures bayesian-uq
#
#   Run QUICK mode (~2 min, ~2 GB RAM):
#     docker run --rm -v $(pwd)/figures:/app/figures bayesian-uq --quick
#
#   If you hit memory limits, increase Docker memory:
#     docker run --rm -m 6g -v $(pwd)/figures:/app/figures bayesian-uq
#
#   Interactive shell inside the container:
#     docker run --rm -it -v $(pwd)/figures:/app/figures \
#       --entrypoint bash bayesian-uq
# ─────────────────────────────────────────────────────────
