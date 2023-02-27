#!/bin/bash
docker run -p 6333:6333 \
    -v $(pwd)/qdrant/storage:/qdrant/storage \
    qdrant/qdrant