FROM ghcr.io/prefix-dev/pixi:latest

WORKDIR /app

# Copy the pixi project definition and install
COPY pixi.toml /app/pixi.toml
RUN pixi install

# Copy application scripts
COPY startup.py /app/startup.py
COPY zmq_tiled_bridge.py /app/zmq_tiled_bridge.py

ENTRYPOINT ["pixi", "run"]
