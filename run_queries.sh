#!/bin/bash

# Path to the file containing the queries
QUERY_FILE="test_queries.txt"
# Output file
OUTPUT_FILE="final_result.csv"

# Ensure the output file is empty or create it
> "$OUTPUT_FILE"

# Read each line from the query file
while IFS= read -r query
do
    # Execute the query and capture the output
    result=$(echo "$query" | docker exec -i 943797a76ffb mgconsole -csv_doublequote --output-format csv)
    
    # Check if the result is empty
    if [[ -z "$result" ]]; then
        result="no_result"
    fi

    # Append the query and its result to the output file
    # Using echo to handle any complex characters that might appear in query or result
    echo "\"$query\"&\"$result\"" >> "$OUTPUT_FILE"

done < "$QUERY_FILE"

echo "Processing complete. Results stored in $OUTPUT_FILE"
