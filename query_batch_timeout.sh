#!/bin/bash

# Check if the correct number of arguments was provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path_to_query_file> <batch_index>"
    exit 1
fi

# Path to the file containing the queries, taken from the first script argument
QUERY_FILE="$1"
BATCH_INDEX="$2"

OUTPUT_DIR="./outputs/${BATCH_INDEX}"
mkdir -p "$OUTPUT_DIR"

# Lock file for atomic counter increments
LOCK_FILE="/tmp/counter.lock"

# Function to safely increment the counter
increment_counter() {
    (
        flock -x 200  # Acquire an exclusive lock
        if [ -f "${OUTPUT_DIR}/counter.txt" ]; then
            counter=$(cat "${OUTPUT_DIR}/counter.txt")
        else
            counter=0
        fi
        counter=$((counter + 1))
        echo $counter > "${OUTPUT_DIR}/counter.txt"
        echo $counter
    ) 200>"$LOCK_FILE"
}

# Read each line from the query file
while IFS= read -r query
do
    if [[ -z "$query" ]]; then
        continue  # Skip empty lines
    fi

    counter=$(increment_counter)

    OUTPUT_FILE="${OUTPUT_DIR}/${counter}.csv"

    # Run the query in the background with a timeout
    timeout 10s docker exec -i 943797a76ffb mgconsole -no_history True --output-format csv -csv_doublequote false <<< "$query" > "$OUTPUT_FILE" &
    DOCKER_PID=$!

    # Wait for the timeout or the process to finish
    wait $DOCKER_PID
    exit_status=$?

    if [ $exit_status -eq 124 ]; then
        # echo "Query execution timed out, killing container process and writing 'timeout fail' to $OUTPUT_FILE"
        docker exec 943797a76ffb pkill -f mgconsole
        echo "timeout fail" > "$OUTPUT_FILE"
    elif [ $exit_status -ne 0 ]; then
        echo "Query execution failed, writing 'failed' to $OUTPUT_FILE"
        echo "failed" > "$OUTPUT_FILE"
    else
        # Check if the output file is empty
        if [ ! -s "$OUTPUT_FILE" ]; then
            # echo "Query $counter returned no results"
            echo "empty" > "$OUTPUT_FILE"
        else
            echo "Query $counter executed successfully, output saved to $OUTPUT_FILE"
        fi
    fi
done < "$QUERY_FILE"

echo "Current Batch queried."