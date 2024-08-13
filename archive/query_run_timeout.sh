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

    # Safely increment the counter and get the current value
    counter=$(increment_counter)

    # Define the output file name based on the counter and batch index
    OUTPUT_FILE="${OUTPUT_DIR}/${counter}.csv"

    # # Execute the query with a 10-second timeout
    # timeout 10s bash -c "echo \"\$query\" | docker exec -i 943797a76ffb mgconsole -no_history True --output-format csv -csv_doublequote false > \"$OUTPUT_FILE\""

    # exit_status=$?

    # Run the Docker command in the background and capture its PID
    echo \"\$query\" | docker exec -i 943797a76ffb mgconsole -no_history True --output-format csv -csv_doublequote false > "$OUTPUT_FILE" &
    DOCKER_PID=$!

    # Use timeout to monitor the process
    timeout 10s bash -c "while kill -0 $DOCKER_PID 2>/dev/null; do sleep 0.1; done"
    exit_status=$?

    if [ -n "$exit_status" ] &&[ $exit_status -eq 124 ]; then
        echo "Query execution timed out, writing 'timeout fail' to $OUTPUT_FILE"
        echo "timeout fail" > "$OUTPUT_FILE"
    elif [ -n "$exit_status" ] &&[ $exit_status -ne 0 ]; then
        echo "Query execution failed, writing 'failed' to $OUTPUT_FILE"
        echo "failed" > "$OUTPUT_FILE"
    else
        echo "$query" | docker exec -i 943797a76ffb mgconsole -no_history True --output-format csv -csv_delimiter ";" -csv_doublequote false > "$OUTPUT_FILE"
        # # Check if the output file is empty
        # if [ ! -s "$OUTPUT_FILE" ]; then
        #     echo "Query $counter returned no results"
        #     echo "empty" > "$OUTPUT_FILE"
        # else
        echo "Query $counter executed successfully, output saved to $OUTPUT_FILE"
    fi
    # fi
    # Increment the counter for the next query
    ((counter++))
done < "$QUERY_FILE"

echo "Current Batch queried."
# Call the Python script to clear the folder
# python clear_folder.py "./outputs"
