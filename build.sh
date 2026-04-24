#!/bin/bash
set -e

cd "$(dirname "$0")"

ENV=${1:-dev}

if [ "$ENV" != "dev" ] && [ "$ENV" != "prod" ]; then
    echo "Usage: ./build.sh [dev|prod]"
    echo "  dev  - Build for development (default)"
    echo "  prod - Build for production"
    exit 1
fi

ENV_FILE=".env.${ENV}"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found"
    exit 1
fi

echo "Building containers for $ENV environment..."
podman-compose --env-file "$ENV_FILE" build

echo ""
echo "Build complete."
echo "Run ./start.sh $ENV to start the services."
