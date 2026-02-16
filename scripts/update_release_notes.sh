#!/bin/bash
#
# Script to help with the NRTK release process. Performs the following steps:
#   - Poetry version (major, minor, or patch)
#   - Combine release note fragments into one file
#   - Update release notes index
#   - Clean pending_release directory
#
# One git commit is created containing the version bump, combined release
# notes, index update, and fragment removal.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
DOCS_DIR="${PROJECT_DIR}/docs"
RELEASE_NOTES_DIR="${DOCS_DIR}/release_notes"
PENDING_RELEASE_NOTES_DIR="${RELEASE_NOTES_DIR}/pending_release"
INDEX_FILE="${RELEASE_NOTES_DIR}/index.rst"

# Check args
if [ "$#" != 1 ]
then
  echo "Please enter valid version bump type. Options: major, minor, or patch"
  exit 1
fi

if [ "$1" != 'major' ] && [ "$1" != 'minor' ] && [ "$1" != 'patch' ]
then
  echo "Please enter valid version bump type. Options: major, minor, or patch"
  exit 1
fi

RELEASE_TYPE="$1"
echo "Release type: ${RELEASE_TYPE}"

# Update version
poetry version "${RELEASE_TYPE}"

# Get version
VERSION="$(poetry version -s)"
VERSION_STR="v${VERSION}"

# Combine release notes
bash "${SCRIPT_DIR}/combine_release_notes.sh" "${VERSION}"
echo "Release notes combined into ${RELEASE_NOTES_DIR}/${VERSION_STR}.rst"

# Add reference to new file in index.rst (insert at top of toctree)
if [ ! -f "$INDEX_FILE" ]; then
  echo "Error: ${INDEX_FILE} not found" >&2
  exit 1
fi
# Insert the new version entry after the ":maxdepth: 1" line followed by a blank line
sed -i "/^   :maxdepth: 1$/,/^$/ {
  /^$/a\\   ${VERSION_STR}
}" "$INDEX_FILE"
echo "Reference added to ${INDEX_FILE}"

# Stage all changes and make a single commit
git add "${PROJECT_DIR}/pyproject.toml"
git add "${RELEASE_NOTES_DIR}/${VERSION_STR}.rst"
git add "${INDEX_FILE}"
git rm "${PENDING_RELEASE_NOTES_DIR}"/*.rst
git commit --no-verify -m "Update to ${VERSION_STR}"
