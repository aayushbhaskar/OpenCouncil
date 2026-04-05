#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/aayushbhaskar/OpenCouncil.git"
INSTALL_DIR="${HOME}/.open-council-app"
VENV_DIR="${INSTALL_DIR}/venv"
BIN_DIR="${HOME}/.local/bin"
EXECUTABLE="${BIN_DIR}/council"

echo "Installing Open Council..."

if [[ -d "${INSTALL_DIR}/.git" ]]; then
  echo "Updating existing installation in ${INSTALL_DIR}"
  git -C "${INSTALL_DIR}" fetch --quiet origin
  git -C "${INSTALL_DIR}" checkout --quiet main
  git -C "${INSTALL_DIR}" pull --quiet --ff-only origin main
else
  echo "Cloning Open Council into ${INSTALL_DIR}"
  rm -rf "${INSTALL_DIR}"
  git clone --quiet "${REPO_URL}" "${INSTALL_DIR}"
fi

echo "Creating virtual environment..."
python3 -m venv "${VENV_DIR}"

echo "Installing package in editable mode..."
"${VENV_DIR}/bin/pip" install --quiet -e "${INSTALL_DIR}"

echo "Linking executable..."
mkdir -p "${BIN_DIR}"
ln -sf "${VENV_DIR}/bin/council" "${EXECUTABLE}"

if [[ ":${PATH}:" != *":${BIN_DIR}:"* ]]; then
  echo
  echo "Add ${BIN_DIR} to PATH to use council globally."
  echo "Run now (current shell):"
  echo "  export PATH=\"${BIN_DIR}:\$PATH\""
  echo
  echo "Or run council directly without PATH:"
  echo "  ${EXECUTABLE} --mode odin"
  echo
  if [[ -n "${SHELL:-}" ]]; then
    case "${SHELL}" in
      */zsh)
        echo "Persist for future sessions:"
        echo "  echo 'export PATH=\"${BIN_DIR}:\$PATH\"' >> ~/.zshrc"
        ;;
      */bash)
        echo "Persist for future sessions:"
        echo "  echo 'export PATH=\"${BIN_DIR}:\$PATH\"' >> ~/.bashrc"
        ;;
      *)
        echo "Persist for your shell by adding this to your shell rc file:"
        echo "  export PATH=\"${BIN_DIR}:\$PATH\""
        ;;
    esac
  fi
fi

echo
echo "Open Council installed."
if [[ ":${PATH}:" == *":${BIN_DIR}:"* ]]; then
  echo "Run: council --mode odin"
else
  echo "Run: ${EXECUTABLE} --mode odin"
fi
