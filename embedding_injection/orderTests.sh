#!/bin/bash
# filepath: /home/aless/Desktop/GliNerBioMed-Label-SoftPrompt/embedding_injection/rank_test_results.sh

# Script to rank test results by Macro F1 (primary) and Micro F1 (secondary)

TEST_DIR="test_results"
OUTPUT_DIR="ranked_results"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Temporary file to store results
TEMP_FILE=$(mktemp)

echo "Extracting metrics from test results..."

# Extract Macro F1 and Micro F1 from each file
for file in "$TEST_DIR"/eval_mlp_prompt_*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        
        # Extract Macro F1 and Micro F1 using grep and awk
        macro_f1=$(grep "Macro F1" "$file" | head -1 | awk -F'|' '{print $3}' | tr -d ' ')
        micro_f1=$(grep "Micro F1" "$file" | head -1 | awk -F'|' '{print $3}' | tr -d ' ')
        
        # Write to temp file: macro_f1 micro_f1 filename
        echo "$macro_f1 $micro_f1 $filename" >> "$TEMP_FILE"
    fi
done

echo "Ranking results..."

# Sort by Macro F1 (descending), then Micro F1 (descending)
sort -k1,1nr -k2,2nr "$TEMP_FILE" > "${TEMP_FILE}_sorted"

# Create ranked copies with numbers, overwriting existing files
rank=1
while read -r macro_f1 micro_f1 filename; do
    # Use rank number as prefix for the filename
    new_filename="${rank}_${filename}"
    
    # Overwrite the existing file in the ranked directory
    cp "$TEST_DIR/$filename" "$OUTPUT_DIR/$new_filename"
    
    echo "Rank $rank: $filename (Macro F1: $macro_f1, Micro F1: $micro_f1)"
    
    ((rank++))
done < "${TEMP_FILE}_sorted"

# Create a summary report
SUMMARY_FILE="$OUTPUT_DIR/00_ranking_summary.md"
echo "# Test Results Ranking" > "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Ranked by Macro F1 (primary) and Micro F1 (secondary)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "| Rank | Filename | Macro F1 | Micro F1 |" >> "$SUMMARY_FILE"
echo "|------|----------|----------|----------|" >> "$SUMMARY_FILE"

rank=1
while read -r macro_f1 micro_f1 filename; do
    padded_rank=$(printf "%02d" $rank)
    echo "| $padded_rank | $filename | $macro_f1 | $micro_f1 |" >> "$SUMMARY_FILE"
    ((rank++))
done < "${TEMP_FILE}_sorted"

# Cleanup
rm "$TEMP_FILE" "${TEMP_FILE}_sorted"

echo ""
echo "✅ Ranking complete!"
echo "📁 Ranked files saved in: $OUTPUT_DIR/"
echo "📊 Summary report: $SUMMARY_FILE"