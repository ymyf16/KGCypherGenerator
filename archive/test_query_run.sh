#!/bin/bash

# Check if an argument was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_query_file>"
    exit 1
fi

# Path to the file containing the queries, taken from the first script argument
QUERY_FILE="$1"

# Counter to keep track of query numbers
counter=1

# Read each line from the query file
while IFS= read -r query
do
    if [[ -z "$query" ]]; then
        continue  # Skip empty lines
    fi

    # Define the output file name based on the counter
    OUTPUT_FILE="./outputs/${counter}.csv"

    # Execute the query and capture the output
    if ! echo "$query" | docker exec -i 943797a76ffb mgconsole -no_history True --output-format csv -csv_delimiter ";" -csv_doublequote false > "$OUTPUT_FILE"; then
        echo "Query execution failed, writing 'failed' to $OUTPUT_FILE"
        echo "failed" > "$OUTPUT_FILE"
    # else
    #     echo "Query $counter executed and output saved to $OUTPUT_FILE"
    fi
    # Increment the counter for the next query
    ((counter++))
done < "$QUERY_FILE"

echo "Current Batch processed."
# Call the Python script to clear the folder
python clear_folder.py "./outputs"