#!/bin/bash
# Monitor training and show validation outputs

echo "=" | head -c 80; echo
echo "📊 MONITORING TRAINING - Validation Outputs"
echo "=" | head -c 80; echo
echo ""
echo "Watching for validation outputs after each epoch..."
echo "Press Ctrl+C to stop monitoring"
echo ""

tail -f training_output.log | grep --line-buffered -A 15 "FIRST 10 VALIDATION OUTPUTS" | while IFS= read -r line; do
    echo "$line"
done

