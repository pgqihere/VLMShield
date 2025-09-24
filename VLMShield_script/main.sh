#!/bin/bash

# Remove set -e to prevent script from exiting on first error
# set -e

# =========================
# Configuration
# =========================
OUTPUT_DIR="../Results/experiment_results"  
CLIP_PATH="/data-c/qipeigui/clip-vit-large-patch14"
CLASSIFIER_PATH="model/vlmshield_classifier.pt"
IMAGE_BASE_DIR=""
BATCH_SIZE=32
DEVICE=""
PYTHON_SCRIPT="VLMShield.py"

# =========================
# Dataset definitions
# =========================
declare -a DATASETS=(
    "datasets/Benign_MM-Vet/data.json|MM-Vet"
    "datasets/Direct_VLSafe/vlsafe_data.json|VLSafe"
    "datasets/Jailbreak_advbench_m/data.json|Advbench_M"
)

# =========================
# Processing
# =========================

echo "VLMShield Batch Processing"
echo "Processing ${#DATASETS[@]} datasets..."
echo "=========================="

mkdir -p "$OUTPUT_DIR"

# Build Python command arguments
python_args="--clip-path \"$CLIP_PATH\" --classifier-path \"$CLASSIFIER_PATH\" --batch-size $BATCH_SIZE"
[[ -n "$IMAGE_BASE_DIR" ]] && python_args="$python_args --image-base-dir \"$IMAGE_BASE_DIR\""
[[ -n "$DEVICE" ]] && python_args="$python_args --device \"$DEVICE\""

# Statistics
total_samples=0
total_processing_time=0
successful_datasets=0
failed_datasets=0

# Record total start time
total_start_time=$(date +%s.%N)

# Process each dataset
for i in "${!DATASETS[@]}"; do
    IFS='|' read -r input_file dataset_name <<< "${DATASETS[$i]}"
    
    temp_output_file="$OUTPUT_DIR/${dataset_name}_temp_results.json"
    final_output_file="$OUTPUT_DIR/${dataset_name}_results.json"
    
    echo
    echo "[$((i+1))/${#DATASETS[@]}] Processing: $dataset_name"
    echo "  Input: $input_file"
    echo "  Output: $final_output_file"
    
    # Check if input file exists
    if [[ ! -f "$input_file" ]]; then
        echo "  ✗ Error: Input file not found!"
        ((failed_datasets++))
        continue
    fi
    
    # Execute command with error handling
    full_command="python \"$PYTHON_SCRIPT\" --input \"$input_file\" --output \"$temp_output_file\" --dataset-name \"$dataset_name\" $python_args"
    
    echo "  Running processing..."
    
    # Use a subshell to capture both success/failure and handle errors gracefully
    if (eval "$full_command") 2>&1; then
        echo "  ✓ Python script completed successfully"
        
        # Check if output file was created
        if [[ -f "$temp_output_file" ]]; then
            echo "  ✓ Output file generated"
            
            # Extract core metrics and create simplified output
            extraction_success=false
            
            if command -v jq &> /dev/null; then
                echo "  Using jq for data extraction..."
                if jq '{
                    dataset_name: .dataset_name,
                    metrics: {
                        asr: .metrics.asr,
                        accuracy: .metrics.accuracy
                    }
                }' "$temp_output_file" > "$final_output_file" 2>/dev/null; then
                    
                    # Extract statistics for summary
                    samples=$(jq -r '.statistics.successfully_processed' "$temp_output_file" 2>/dev/null || echo "0")
                    processing_time=$(jq -r '.statistics.total_processing_time_seconds' "$temp_output_file" 2>/dev/null || echo "0")
                    extraction_success=true
                fi
            fi
            
            # Fallback to Python if jq failed or not available
            if [[ "$extraction_success" = false ]]; then
                echo "  Using Python for data extraction..."
                python_extraction=$(python3 -c "
import json
import sys

try:
    with open('$temp_output_file', 'r') as f:
        data = json.load(f)
    
    simplified = {
        'dataset_name': data['dataset_name'],
        'metrics': {
            'asr': data['metrics']['asr'],
            'accuracy': data['metrics']['accuracy']
        }
    }
    
    with open('$final_output_file', 'w') as f:
        json.dump(simplified, f, indent=2)
    
    print('SUCCESS')
    print(data['statistics']['successfully_processed'])
    print(data['statistics']['total_processing_time_seconds'])
except Exception as e:
    print('ERROR')
    print('0')
    print('0')
" 2>/dev/null)
                
                result_status=$(echo "$python_extraction" | head -n1)
                if [[ "$result_status" = "SUCCESS" ]]; then
                    samples=$(echo "$python_extraction" | sed -n '2p')
                    processing_time=$(echo "$python_extraction" | sed -n '3p')
                    extraction_success=true
                fi
            fi
            
            # Remove temporary file
            rm -f "$temp_output_file"
            
            if [[ "$extraction_success" = true ]]; then
                # Update totals
                total_samples=$((total_samples + samples))
                total_processing_time=$(echo "$total_processing_time + $processing_time" | bc -l 2>/dev/null || echo "$total_processing_time")
                ((successful_datasets++))
                
                echo "  ✓ Completed successfully - Samples: $samples, Time: ${processing_time}s"
            else
                echo "  ✗ Failed to extract metrics from output file"
                ((failed_datasets++))
            fi
        else
            echo "  ✗ Failed - Output file not generated"
            ((failed_datasets++))
        fi
    else
        echo "  ✗ Failed - Command execution error"
        ((failed_datasets++))
        # Clean up any partial temp file
        rm -f "$temp_output_file"
    fi
    
    echo "  Current progress: $successful_datasets successful, $failed_datasets failed"
done

echo
echo "=========================="
echo "All datasets processing completed!"
echo "=========================="

# Record total end time
total_end_time=$(date +%s.%N)
total_wall_time=$(echo "$total_end_time - $total_start_time" | bc -l 2>/dev/null || echo "0")

# Calculate average time per sample
if [[ $total_samples -gt 0 ]]; then
    avg_time_per_sample=$(echo "scale=6; $total_processing_time / $total_samples" | bc -l 2>/dev/null || echo "0")
else
    avg_time_per_sample=0
fi

# Display final summary
echo "Final Summary:"
echo "  Total datasets: ${#DATASETS[@]}"
echo "  Successful: $successful_datasets"
echo "  Failed: $failed_datasets"
echo "  Total samples: $total_samples"
echo "  Total processing time: ${total_processing_time}s"
echo "  Average time per sample: ${avg_time_per_sample}s"

# Create final summary with error handling
summary_file="$OUTPUT_DIR/summary.json"
echo
echo "Creating summary file: $summary_file"

cat > "$summary_file" << EOF
{
  "summary": {
    "timestamp": "$(date -Iseconds)",
    "total_datasets": ${#DATASETS[@]},
    "successful_datasets": $successful_datasets,
    "failed_datasets": $failed_datasets,
    "total_samples_processed": $total_samples,
    "total_processing_time_seconds": $total_processing_time,
    "total_wall_time_seconds": $total_wall_time,
    "average_time_per_sample_seconds": $avg_time_per_sample,
    "datasets": [
EOF

# Add individual dataset results to summary
first=true
for i in "${!DATASETS[@]}"; do
    IFS='|' read -r input_file dataset_name <<< "${DATASETS[$i]}"
    result_file="$OUTPUT_DIR/${dataset_name}_results.json"
    
    if [[ -f "$result_file" ]]; then
        if [[ "$first" = true ]]; then
            first=false
        else
            echo "," >> "$summary_file"
        fi
        
        # Add the content without modifying
        cat "$result_file" >> "$summary_file"
    fi
done

cat >> "$summary_file" << EOF

    ]
  }
}
EOF

echo "✓ Summary file created successfully"
echo
echo "All results saved to: $OUTPUT_DIR"
echo "Main summary: $summary_file"
echo "=========================="