#!/bin/bash

# Path to the file containing the queries
QUERY_FILE="/Users/yufeimeng/Desktop/KGCypherGenerator/output_queries.txt"

# Counter to keep track of query numbers
counter=1

# Read each line from the query file
while IFS= read -r query
do
    if [[ -z "$query" ]]; then
        continue  # Skip empty lines
    fi

    # Define the output file name based on the counter
    OUTPUT_FILE="./outputs/query${counter}.csv"

    # Execute the query and capture the output
    echo "$query" | docker exec -i 943797a76ffb mgconsole --output-format csv -csv_delimiter ";" -csv_doublequote true > "$OUTPUT_FILE"
    
    echo "Query $counter executed and output saved to $OUTPUT_FILE"

    # Increment the counter for the next query
    ((counter++))
done < "$QUERY_FILE"

echo "All queries processed."
