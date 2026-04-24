#!/bin/bash
set -e

cd "$(dirname "$0")"

ENV=${1:-dev}

if [ "$ENV" != "dev" ] && [ "$ENV" != "prod" ]; then
    echo "Usage: ./stop.sh [dev|prod]"
    echo "  dev  - Stop development environment (default)"
    echo "  prod - Stop production environment"
    exit 1
fi

ENV_FILE=".env.${ENV}"

echo "Stopping services ($ENV)..."
podman-compose --env-file "$ENV_FILE" down

echo "Services stopped."
