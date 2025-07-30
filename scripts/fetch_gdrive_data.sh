#!/usr/bin/env bash
# Fetch test data from Google Drive with integrity check
set -euo pipefail

# Configuration from environment
FILE_ID="${GDRIVE_FILE_ID:-}"
SHA_EXPECT="${GDRIVE_SHA256:-}"
TARGET="tests/_data/sample.zip"
DATA_DIR="tests/_data"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}[gdrive]${NC} Checking for test data..."

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Check if already downloaded
if [[ -f "$TARGET" ]]; then
    echo -e "${GREEN}[gdrive]${NC} Cache hit - data already present"
    # Verify integrity if SHA provided
    if [[ -n "$SHA_EXPECT" ]]; then
        echo "$SHA_EXPECT  $TARGET" | sha256sum -c - >/dev/null 2>&1 || {
            echo -e "${YELLOW}[gdrive]${NC} Checksum mismatch, re-downloading..."
            rm -f "$TARGET"
        }
    else
        exit 0
    fi
fi

# Check required variables
if [[ -z "$FILE_ID" ]]; then
    echo -e "${YELLOW}[gdrive]${NC} GDRIVE_FILE_ID not set, skipping download"
    exit 0
fi

# Install gdown if not present
if ! command -v gdown &> /dev/null; then
    echo -e "${YELLOW}[gdrive]${NC} Installing gdown..."
    pip install --quiet gdown
fi

# Download file
echo -e "${YELLOW}[gdrive]${NC} Downloading test data from Google Drive..."
gdown --id "$FILE_ID" -O "$TARGET" --quiet

# Verify checksum if provided
if [[ -n "$SHA_EXPECT" ]]; then
    echo -e "${YELLOW}[gdrive]${NC} Verifying checksum..."
    echo "$SHA_EXPECT  $TARGET" | sha256sum -c - || {
        echo -e "${YELLOW}[gdrive]${NC} Checksum verification failed!"
        rm -f "$TARGET"
        exit 1
    }
fi

# Extract if it's a zip file
if [[ "$TARGET" == *.zip ]]; then
    echo -e "${YELLOW}[gdrive]${NC} Extracting archive..."
    unzip -oq "$TARGET" -d "$DATA_DIR"
fi

echo -e "${GREEN}[gdrive]${NC} Test data ready ✓"