#!/bin/bash

# Run the Docker container
docker run -d \
  --name crossword-container \
  -p 3000:3000 \
  --env-file .env \
  -v "$(pwd)":/app \
  crossword-app
