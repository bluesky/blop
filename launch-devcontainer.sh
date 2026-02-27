#!/bin/bash

# Simple script to launch the devcontainer with OpenCode

echo "Starting devcontainer..."
docker compose up -d

echo ""
echo "Entering container..."
docker compose exec devcontainer bash -c "source /root/.bashrc && exec bash"
