#!/bin/bash
cd "$(dirname "$0")"

PYTHON=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done

if [ -z "$PYTHON" ]; then
  echo
  echo "Python 3 could not be found on this computer."
  echo "Please install Python 3 from https://www.python.org/downloads/ and try again."
  echo
  read -n 1 -s -r -p "Press any key to close this window."
  exit 1
fi

if ! "$PYTHON" -c "import flask, pandas, sklearn, shap, pyarrow, joblib" >/dev/null 2>&1; then
  echo
  echo "Preparing the application (first run only, this may take a few minutes)..."
  echo
  "$PYTHON" -m pip install --quiet --disable-pip-version-check -r requirements.txt
fi

if ! "$PYTHON" -c "import flask, pandas, sklearn, shap, pyarrow, joblib" >/dev/null 2>&1; then
  echo
  echo "The application could not install its required components."
  echo "Please check your internet connection and try again."
  echo
  read -n 1 -s -r -p "Press any key to close this window."
  exit 1
fi

exec "$PYTHON" app.py
