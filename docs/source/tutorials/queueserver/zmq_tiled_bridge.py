"""
ZMQ-to-Tiled bridge.

Subscribes to the ZMQ proxy output and writes all Bluesky documents to a Tiled server.
This ensures that data produced by the queueserver is persisted and queryable.
"""

import os

from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky_tiled_plugins import TiledWriter
from tiled.client import from_uri

zmq_addr = os.environ.get("ZMQ_ADDR", "tcp://localhost:5578")
tiled_uri = os.environ.get("TILED_URI", "http://localhost:8000")
tiled_api_key = os.environ.get("TILED_API_KEY", "tutorial_key")

print(f"[BRIDGE] Connecting to Tiled at {tiled_uri}")
tiled_client = from_uri(tiled_uri, api_key=tiled_api_key)

print(f"[BRIDGE] Subscribing to ZMQ at {zmq_addr}")
# Parse address into (host, port) tuple for RemoteDispatcher
_addr = zmq_addr.replace("tcp://", "").split(":")
dispatcher = RemoteDispatcher((_addr[0], int(_addr[1])))

tiled_writer = TiledWriter(tiled_client)
dispatcher.subscribe(tiled_writer)

print("[BRIDGE] Starting dispatcher (blocking)...")
dispatcher.start()
