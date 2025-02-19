#!/bin/bash

# ./batch_convert_xls_to_csv.sh 241107_16_17_PNPO_PTZ/241107_16_17_PNPO_PTZ_rawoutput

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

# Loop through all .xls files in the input folder
for FILE in "$INPUT_FOLDER"/*.xls; do
  # Check if there are any .xls files
  if [ ! -e "$FILE" ]; then
    echo "No .xls files found in $INPUT_FOLDER"
    exit 1
  fi

  # Get the base name of the file (without path or extension)
  BASENAME=$(basename "$FILE" .xls)

  # Define the output .csv file path
  OUTPUT_FILE="$OUTPUT_FOLDER/${BASENAME}.csv"

  # Convert the file using ssconvert
  echo "Converting $FILE to $OUTPUT_FILE"
  ssconvert "$FILE" "$OUTPUT_FILE"
done

echo "Conversion complete! CSV files are in $OUTPUT_FOLDER"
