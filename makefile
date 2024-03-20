# Makefile for MLOps-task4 repository

# Define variables
PYTHON := python3

# Default target
.DEFAULT_GOAL := help

# Run the script
run:
    @echo "Running the script..."
    $(PYTHON) DeepModel.ipynb
    @echo "Script execution complete."

# Clean up generated files
clean:
    @echo "Cleaning up..."
    # Remove any temporary or generated files
    rm -rf __pycache__  # Remove Python cache files
    # Add any other clean-up commands as necessary
    @echo "Clean up complete."

# Help
help:
    @echo "Available targets:"
    @echo "  run     : Run the script"
    @echo "  clean   : Clean up generated files"
    @echo "  help    : Show this help message"
