#!/bin/bash

# Setup script for PEBBLE Preference Collection UI
# Authors: Yang Guo & Marzieh Ghayour

echo "=================================="
echo "PEBBLE UI Setup"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt --break-system-packages

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p static/videos
mkdir -p static/css
mkdir -p static/js
mkdir -p templates
mkdir -p trajectory_data
mkdir -p preference_data

echo "✓ Directories created"

# Generate demo videos (optional)
echo ""
read -p "Generate demo trajectory videos? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Generating demo videos..."
    python3 trajectory_video_generator.py
    echo "✓ Demo videos generated"
fi

# Check if all files exist
echo ""
echo "Checking required files..."

FILES=(
    "app.py"
    "templates/index.html"
    "static/css/style.css"
    "static/js/app.js"
    "trajectory_video_generator.py"
    "README.md"
)

ALL_EXIST=true
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file MISSING"
        ALL_EXIST=false
    fi
done

if [ "$ALL_EXIST" = true ]; then
    echo ""
    echo "=================================="
    echo "✓ Setup Complete!"
    echo "=================================="
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Configure participant in app.py:"
    echo "   - Set participant_id (e.g., 'P01')"
    echo "   - Set session_id (e.g., 1)"
    echo "   - Set condition (e.g., 'baseline')"
    echo ""
    echo "2. Start the server:"
    echo "   python3 app.py"
    echo ""
    echo "3. Open in browser:"
    echo "   http://localhost:5000"
    echo ""
    echo "4. Data will be saved to:"
    echo "   preference_data/"
    echo ""
else
    echo ""
    echo "⚠️  Some files are missing!"
    echo "Please ensure all files are in the correct locations."
fi
