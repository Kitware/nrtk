#!/bin/bash

# Script to combine pending release note fragments into a single release
# notes file.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
DOCS_DIR="${PROJECT_DIR}/docs"
RELEASE_NOTES_DIR="${DOCS_DIR}/release_notes"

usage="Usage: $0 <version-string>"

if [ "$#" -ne 1 ]; then
    echo "$usage" >&2
    exit 1
fi

version="$1"
version_str="v${version}"
OUTPUT_FILE="${RELEASE_NOTES_DIR}/${version_str}.rst"
PENDING_RELEASE_NOTES_DIR="${RELEASE_NOTES_DIR}/pending_release"

if [ ! -d "$PENDING_RELEASE_NOTES_DIR" ]; then
    echo "Error: Directory '$PENDING_RELEASE_NOTES_DIR' does not exist." >&2
    exit 1
fi

# Check that there are fragment files to combine (ignore .gitkeep)
shopt -s nullglob
fragments=("$PENDING_RELEASE_NOTES_DIR"/*.rst)
shopt -u nullglob

if [ ${#fragments[@]} -eq 0 ]; then
    echo "Error: No .rst fragments found in '$PENDING_RELEASE_NOTES_DIR'." >&2
    exit 1
fi

# Write header
{
    echo "$version_str"
    echo "${version_str//?/=}"
    echo
} > "$OUTPUT_FILE"

# Concatenate all fragments, separated by blank lines
first=true
for file in "${fragments[@]}"; do
    if [ "$first" = true ]; then
        first=false
    else
        echo >> "$OUTPUT_FILE"
    fi
    cat "$file" >> "$OUTPUT_FILE"
done

# Ensure file ends with a single newline
printf '%s\n' "$(cat "$OUTPUT_FILE")" > "$OUTPUT_FILE"

echo "Release notes generated: $OUTPUT_FILE"
