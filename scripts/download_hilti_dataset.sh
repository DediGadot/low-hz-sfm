#!/bin/bash
################################################################################
# Download Hilti SLAM Challenge 2023 Dataset
#
# This script downloads ROS bag files from the Hilti SLAM Challenge 2023 dataset
# using wget with automatic resume capability.
#
# Dependencies:
# - wget (pre-installed on most Linux systems)
#
# Usage:
#   bash scripts/download_hilti_dataset.sh
#
#   Or make it executable and run:
#   chmod +x scripts/download_hilti_dataset.sh
#   ./scripts/download_hilti_dataset.sh
#
# Sample Input:
#   User selects download option interactively
#
# Expected Output:
#   Downloaded ROS bag files in datasets/hilti/rosbags/
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Base URL
BASE_URL="https://tp-public-facing.s3.eu-north-1.amazonaws.com/Challenges/2023"

# Determine output directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/datasets/hilti/rosbags"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print header
echo "================================================================================"
echo "  Hilti SLAM Challenge 2023 Dataset Downloader"
echo "================================================================================"
echo ""

# Download function
download_file() {
    local filename=$1
    local size=$2
    local desc=$3
    local url="$BASE_URL/$filename"
    local output_path="$OUTPUT_DIR/$filename"

    echo -e "${BLUE}📥 Downloading:${NC} $filename ($size GB) - $desc"

    # Use wget with resume capability (-c), show progress bar
    if wget -c -q --show-progress "$url" -O "$output_path"; then
        echo -e "${GREEN}✅ Downloaded:${NC} $filename"
        return 0
    else
        echo -e "${RED}❌ Failed:${NC} $filename"
        return 1
    fi
}

# Menu function
show_menu() {
    echo ""
    echo "Preset Options:"
    echo "  1. Site 1 All (Multi-visit, recommended) - 5 files, 107.2 GB"
    echo "  2. Site 1 First 3 (Quick start) - 3 files, 58.2 GB"
    echo "  3. Site 2 Robot - 3 files, 123.5 GB"
    echo "  4. Site 2 Handheld - 3 files, 43.7 GB"
    echo "  5. Site 3 All - 4 files, 53.6 GB"
    echo "  6. Download ALL sequences - 15 files, 328.0 GB ⚠️"
    echo "  0. Exit"
    echo "================================================================================"
    echo ""
}

# Download Site 1 (all)
download_site1_all() {
    echo -e "\n${BLUE}Selected: Site 1 All (5 files, 107.2 GB)${NC}\n"

    download_file "site1_handheld_1.bag" "22.0" "Floor 0"
    download_file "site1_handheld_2.bag" "17.9" "Floor 1"
    download_file "site1_handheld_3.bag" "18.3" "Floor 2"
    download_file "site1_handheld_4.bag" "31.9" "Underground"
    download_file "site1_handheld_5.bag" "17.1" "Stairs"
}

# Download Site 1 (first 3)
download_site1_first3() {
    echo -e "\n${BLUE}Selected: Site 1 First 3 (3 files, 58.2 GB)${NC}\n"

    download_file "site1_handheld_1.bag" "22.0" "Floor 0"
    download_file "site1_handheld_2.bag" "17.9" "Floor 1"
    download_file "site1_handheld_3.bag" "18.3" "Floor 2"
}

# Download Site 2 Robot
download_site2_robot() {
    echo -e "\n${BLUE}Selected: Site 2 Robot (3 files, 123.5 GB)${NC}\n"

    download_file "site2_robot_1.bag" "63.3" "Parking (3 floors)"
    download_file "site2_robot_2.bag" "27.7" "Floor 1 Large room"
    download_file "site2_robot_3.bag" "32.5" "Floor 2 Large room"
}

# Download Site 2 Handheld
download_site2_handheld() {
    echo -e "\n${BLUE}Selected: Site 2 Handheld (3 files, 43.7 GB)${NC}\n"

    download_file "site2_handheld_4.bag" "9.3" "Central staircase"
    download_file "site2_handheld_5.bag" "20.6" "Vault Staircase"
    download_file "site2_handheld_6.bag" "13.8" "Large room connector"
}

# Download Site 3 All
download_site3_all() {
    echo -e "\n${BLUE}Selected: Site 3 All (4 files, 53.6 GB)${NC}\n"

    download_file "site3_handheld_1.bag" "9.7" "Underground 1"
    download_file "site3_handheld_2.bag" "14.6" "Underground 2"
    download_file "site3_handheld_3.bag" "18.7" "Underground 3"
    download_file "site3_handheld_4.bag" "10.6" "Underground 4"
}

# Download all sequences
download_all() {
    echo -e "\n${YELLOW}⚠️  This will download 328 GB. Continue? (yes/no):${NC} "
    read -r confirm

    if [ "$confirm" != "yes" ]; then
        echo -e "${RED}❌ Download cancelled.${NC}"
        exit 0
    fi

    echo -e "\n${BLUE}Selected: ALL sequences (15 files, 328.0 GB)${NC}\n"

    # Site 1
    download_file "site1_handheld_1.bag" "22.0" "Site 1 - Floor 0"
    download_file "site1_handheld_2.bag" "17.9" "Site 1 - Floor 1"
    download_file "site1_handheld_3.bag" "18.3" "Site 1 - Floor 2"
    download_file "site1_handheld_4.bag" "31.9" "Site 1 - Underground"
    download_file "site1_handheld_5.bag" "17.1" "Site 1 - Stairs"

    # Site 2 Robot
    download_file "site2_robot_1.bag" "63.3" "Site 2 Robot - Parking"
    download_file "site2_robot_2.bag" "27.7" "Site 2 Robot - Floor 1"
    download_file "site2_robot_3.bag" "32.5" "Site 2 Robot - Floor 2"

    # Site 2 Handheld
    download_file "site2_handheld_4.bag" "9.3" "Site 2 - Central staircase"
    download_file "site2_handheld_5.bag" "20.6" "Site 2 - Vault Staircase"
    download_file "site2_handheld_6.bag" "13.8" "Site 2 - Large room connector"

    # Site 3
    download_file "site3_handheld_1.bag" "9.7" "Site 3 - Underground 1"
    download_file "site3_handheld_2.bag" "14.6" "Site 3 - Underground 2"
    download_file "site3_handheld_3.bag" "18.7" "Site 3 - Underground 3"
    download_file "site3_handheld_4.bag" "10.6" "Site 3 - Underground 4"
}

# Main execution
echo -e "${BLUE}📁 Output directory:${NC} $OUTPUT_DIR"

show_menu

echo -n "Enter your choice (0-6): "
read -r choice

case $choice in
    1)
        download_site1_all
        ;;
    2)
        download_site1_first3
        ;;
    3)
        download_site2_robot
        ;;
    4)
        download_site2_handheld
        ;;
    5)
        download_site3_all
        ;;
    6)
        download_all
        ;;
    0)
        echo -e "${YELLOW}👋 Goodbye!${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}❌ Invalid choice. Please run again.${NC}"
        exit 1
        ;;
esac

# Summary
echo ""
echo "================================================================================"
echo -e "${GREEN}✅ Download complete!${NC}"
echo "================================================================================"
echo -e "${BLUE}📁 Files saved to:${NC} $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Extract frames:"
echo "     uv run python -m sfm_experiments.cli extract-frames \\"
echo "         $OUTPUT_DIR/site1_handheld_1.bag \\"
echo "         datasets/hilti/frames/sequence_01 \\"
echo "         --fps 0.25"
echo ""
echo "  2. Update config (if needed):"
echo "     Edit configs/hilti.yaml to match downloaded files"
echo ""
echo "  3. Run experiment:"
echo "     uv run python -m sfm_experiments.cli run-experiment \\"
echo "         --config-file configs/hilti.yaml \\"
echo "         --output-dir results \\"
echo "         --visits '1,2,3'"
echo ""
echo "💡 Tip: Interrupted downloads will resume automatically if you run the script again"
echo "================================================================================"
