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

    # Run the Docker command in the background and capture its PID
    echo \"$query\" | docker exec -i 943797a76ffb mgconsole -no_history True --output-format csv -csv_doublequote false > "$OUTPUT_FILE" &
    DOCKER_PID=$!

    # Initiate timeout and process monitoring
    timeout_duration=10
    start_time=$(date +%s)

    # Loop to monitor and handle timeout
    while : ; do
        current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))

        if [ "$elapsed_time" -ge "$timeout_duration" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Process exceeded $timeout_duration seconds, checking running containers..." >> "$OUTPUT_DIR/log.txt"
            echo "timeout fail" > "$OUTPUT_FILE"

            # Using Docker command to inspect all containers for the specific PID
            for i in $(docker container ls --format "{{.ID}}"); do
                read container_pid container_name <<< $(docker inspect -f '{{.State.Pid}} {{.Name}}' $i)
                if [ "$container_pid" -eq "$DOCKER_PID" ]; then
                    echo "$(date '+%Y-%m-%d %H:%M:%S') - Found the running process in container $container_name, terminating..." >> "$OUTPUT_DIR/log.txt"
                    docker kill $i
                    echo "timeout fail" > "$OUTPUT_FILE"
                    echo "$(date '+%Y-%m-%d %H:%M:%S') - Timeout fail written to $OUTPUT_FILE" >> "$OUTPUT_DIR/log.txt"
                    break 2
                fi
            done
            echo "$(date '+%Y-%m-%d %H:%M:%S') - No specific process found, writing timeout fail..." >> "$OUTPUT_DIR/log.txt"
            echo "timeout fail" > "$OUTPUT_FILE"
        fi

        if ! kill -0 $DOCKER_PID 2>/dev/null; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Process terminated within the timeout." >> "$OUTPUT_DIR/log.txt"
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Query $counter executed successfully, output saved to $OUTPUT_FILE" >> "$OUTPUT_DIR/log.txt"
            break
        fi
        sleep 0.1
    done
done < "$QUERY_FILE"
echo "Current Batch queried."
