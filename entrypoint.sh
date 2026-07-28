#!/bin/sh

# Function to handle conditional installation logic
install_pkg() {
    PKG_NAME=$1
    VERSION=$2

    # Case 1: Skip if version is 'None' or empty string
    if [ "$VERSION" = "None" ] || [ -z "$VERSION" ]; then
        echo "[INFO] Skipping $PKG_NAME (Version set to None)."
        return 0
    fi

    # Case 2: Install latest if version is 'latest'
    if [ "$VERSION" = "latest" ]; then
        echo "[INFO] Installing latest version of $PKG_NAME..."
        pip install "$PKG_NAME"
    
    # Case 3: Install specific version
    else
        echo "[INFO] Installing $PKG_NAME==$VERSION..."
        pip install "${PKG_NAME}==${VERSION}"
    fi
}

echo "--- Starting Container Bootstrapping ---"

# Perform installations based on environment variables
install_pkg "scikit-learn" "$SCIKIT_LEARN_VERSION"
install_pkg "torch"        "$PYTORCH_VERSION"
install_pkg "tensorflow"   "$TENSORFLOW_VERSION"

echo "--- Bootstrapping Complete. Launching App ---"

# start the application with main.py
exec python main.py
