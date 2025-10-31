#!/bin/bash
# Build LaTeX thesis and convert to text for review

set -e  # Exit on error

echo "========================================="
echo "Building LaTeX thesis..."
echo "========================================="

cd /home/matt/Dev/Matt-K-MSc-AI-Thesis
latexmk -g -output-directory=build main.tex

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Build successful! Converting PDF to text..."
    echo "========================================="
    
    pdftotext -layout build/main.pdf build/main.txt
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "========================================="
        echo "Success! Files created:"
        echo "  - build/main.pdf"
        echo "  - build/main.txt"
        echo "========================================="
    else
        echo ""
        echo "ERROR: PDF to text conversion failed"
        exit 1
    fi
else
    echo ""
    echo "ERROR: LaTeX build failed"
    exit 1
fi

