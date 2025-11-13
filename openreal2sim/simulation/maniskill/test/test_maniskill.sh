#!/bin/bash
# Convenience script to test ManiSkill integration

set -e

echo "=================================="
echo "OpenReal2Sim ManiSkill Test Script"
echo "=================================="
echo ""

# # Check if we're in the right directory
# if [ ! -d "openreal2sim/simulation/maniskill" ]; then
#     echo "Error: Please run this script from the OpenReal2Sim root directory"
#     exit 1
# fi

# Default scene
SCENE="${1:-/mnt/storage/projects/haoyang/OpenReal2Sim/outputs/demo_genvideo/scene/scene.json}"

echo "Testing with scene: $SCENE"
echo ""

# Check if scene exists
if [ ! -f "$SCENE" ]; then
    echo "Error: Scene not found: $SCENE"
    echo ""
    echo "Available scenes:"
    find outputs -name "scene.json" 2>/dev/null || echo "  No scenes found in outputs/"
    exit 1
fi

# Check dependencies
# echo "Checking dependencies..."
# python -c "import mani_skill" 2>/dev/null || {
#     echo "Warning: mani_skill not installed. Installing..."
#     pip install -r openreal2sim/simulation/maniskill/requirements.txt
# }

echo ""
echo "Running tests..."
echo "=================================="

# Run the test
python /mnt/storage/projects/haoyang/OpenReal2Sim/openreal2sim/simulation/maniskill/test_env.py --scene "$SCENE"

echo ""
echo "=================================="
echo "Test complete!"
echo ""
echo "To run with a different scene:"
echo "  ./scripts/test_maniskill.sh outputs/demo_image/scene/scene.json"
echo ""
echo "To run the example:"
echo "  python openreal2sim/simulation/maniskill/example.py"

