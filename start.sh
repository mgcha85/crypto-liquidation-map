#!/bin/bash
set -e

cd "$(dirname "$0")"

ENV=${1:-dev}

if [ "$ENV" != "dev" ] && [ "$ENV" != "prod" ]; then
    echo "Usage: ./start.sh [dev|prod]"
    echo "  dev  - Start development environment (default)"
    echo "  prod - Start production environment"
    exit 1
fi

ENV_FILE=".env.${ENV}"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found"
    echo "Copy .env.example to $ENV_FILE and configure it."
    exit 1
fi

echo "Starting services ($ENV)..."
podman-compose --env-file "$ENV_FILE" up -d

echo ""
echo "Services started ($ENV):"
echo "  - Trader API: http://localhost:8080"
echo "  - Dashboard:  http://localhost:3000"
echo ""
echo "View logs: podman-compose --env-file $ENV_FILE logs -f"
