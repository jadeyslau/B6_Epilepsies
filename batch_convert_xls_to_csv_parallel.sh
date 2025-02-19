#!/bin/bash

# Ensure the input folder is provided as an argument
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <path_to_folder_with_xls_files>"
  exit 1
fi

# Set variables
INPUT_FOLDER="$1"
OUTPUT_FOLDER="${INPUT_FOLDER}/raw_converted_csv"

# Create the output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Check if there are any .xls files
if ! ls "$INPUT_FOLDER"/*.xls &>/dev/null; then
  echo "No .xls files found in $INPUT_FOLDER"
  exit 1
fi

# Export variables for GNU Parallel
export OUTPUT_FOLDER

# Parallelize conversion using GNU Parallel
find "$INPUT_FOLDER" -name "*.xls" | parallel -j "$(nproc)" '
  BASENAME=$(basename "{}" .xls)
  OUTPUT_FILE="$OUTPUT_FOLDER/${BASENAME}.csv"
  if [ -f "$OUTPUT_FILE" ]; then
    echo "Skipping {} as $OUTPUT_FILE already exists"
  else
    echo "Converting {} to $OUTPUT_FILE"
    ssconvert "{}" "$OUTPUT_FILE" && rm -f "{}"
  fi
'

echo "Conversion complete! CSV files are in $OUTPUT_FOLDER"
