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

# Create outputs directory if it doesn't exist
mkdir -p ./outputs

# Read each line from the query file
while IFS= read -r query
do
    if [[ -z "$query" ]]; then
        continue  # Skip empty lines
    fi

    # Define a temporary file name
    TEMP_FILE=$(mktemp)

    # Execute the query and capture the output to the temporary file
    if echo "$query" | docker exec -i 943797a76ffb mgconsole -no_history True --output-format csv -csv_delimiter ";" -csv_doublequote false > "$TEMP_FILE"; then
        # Check if the file is not empty (has content beyond the header)
        if [ $(wc -l < "$TEMP_FILE") -gt 1 ]; then
            # If file has content, move it to the outputs directory with the correct name
            mv "$TEMP_FILE" "./outputs/${counter}.csv"
            echo "Query $counter executed and output saved to ./outputs/${counter}.csv"
        else
            # If file is empty or only has a header, delete it
            rm "$TEMP_FILE"
            # echo "Query $counter returned no results, output not saved"
        fi
    else
        echo "Query $counter execution failed"
        rm "$TEMP_FILE"
    fi

    # Increment the counter for the next query
    ((counter++))
done < "$QUERY_FILE"

echo "Current Batch processed."