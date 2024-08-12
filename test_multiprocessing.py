import shutil
import pandas as pd
import os
import csv
import subprocess
import logging
import re
import mgclient
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
conn = mgclient.connect(host='127.0.0.1', port=7687)
cursor = conn.cursor()

def write_queries_to_file(queries, filename):
    """Writes a list of queries to a specified file."""
    with open(filename, 'w') as file:
        for query in queries:
            file.write(query + '\n')
            
def aggregate_text_files(folder_path, output_file):
    # Initialize a list to store the aggregated data
    aggregated_data = []

    # List all files in the specified folder
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]  # or change to '.txt' if appropriate
    # file_list.sort()
    # print(file_list)
    file_list = sorted(file_list, key=lambda x: int(re.search(r'\d+', x).group()))
    # print(test)

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        # Check if the file is empty
        if os.stat(file_path).st_size == 0:
            aggregated_data.append({"variables_returned": "no_result", "results_returned": "no_result"})
            continue
        
        # Open and read the file as a plain text file
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # Check again for empty content after reading lines
        if not lines:
            aggregated_data.append({"variables_returned": "no_result", "results_returned": "no_result"})
            continue

        # First line as 'variables_returned'
        variables_returned = lines[0].strip()
        if variables_returned == 'failed':
            results_returned = "failed"
        else:
        # Join the rest of the lines for 'results_returned'
            if len(lines) > 1:
                results_returned = '; '.join([line.strip().replace('\n', ';') for line in lines[1:]])
            else:
                results_returned = "no_result"
        
        # aggregated_data.append([variables_returned, results_returned])
        # with open(output_file, 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(aggregated_data)
        # print("Data written to CSV file successfully.")

        aggregated_data.append({
            "variables_returned": variables_returned,
            "results_returned": results_returned
        })
    # Create a DataFrame from the aggregated data
    result_df = pd.DataFrame(aggregated_data)
    # Save the aggregated data to a CSV file
    result_df.to_csv(output_file, index=False)

def merge_csv_files(directory, output_file):
    # Path to the directory containing CSV files
    path = directory
    # List to hold DataFrames
    dfs = []
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    # Sort the files to maintain order - ensure this sorts as you need (e.g., numerically, alphabetically)
    # csv_files.sort()
    csv_files = sorted(csv_files, key=lambda x: int(re.search(r'\d+', x).group()))
    # Read each CSV file and store it in the list
    for file in csv_files:
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        dfs.append(df)
    # Concatenate all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)

    # Optionally, reset the index to make it an explicit column in the DataFrame
    merged_df.reset_index(inplace=True, drop=False)
    merged_df.rename(columns={'index': 'ID'}, inplace=True)

    # Write the concatenated DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)
    print(f'Merged CSV saved as {output_file}')

# def batch_processing(query_list, batch_size=10):
#     total_queries = len(query_list)
#     # batch_size = 10
#     num_batches = total_queries // batch_size
#     # Make sure the script is executable
#     subprocess.run(['chmod', '+x', 'test_query_run.sh'])  # Move outside the loop
#     logging.basicConfig(level=logging.INFO)

#     # Processing each batch
#     counter = 0
#     # while counter < num_batches:
#     for counter in range(num_batches):
#         logging.info("Processing batch %d", counter)
#         start_index = counter * batch_size
#         end_index = start_index + batch_size
#         current_batch = query_list[start_index:end_index]
#         query_filename = f"./input_batch/batch{counter}.txt"
#         # Write current batch to file and execute it
#         write_queries_to_file(current_batch, query_filename)
#         # subprocess.run(['chmod', '+x', 'test_query_run.sh'])
#         # Execute the script with the query file as an argument
#         subprocess.run(['./test_query_run.sh', query_filename])

#         folder_path = './outputs'
#         output_file = f'./aggregates/results{counter}.csv'
#         aggregate_text_files(folder_path, output_file)
#         # clear_directory(folder_path)
#         counter += 1

# def execute_query_script(query_filename, output_file):
#     try:
#         # Set timeout for subprocess to 10 seconds
#         subprocess.run(['./test_query_run.sh', query_filename], timeout=10, check=True)
#     except subprocess.TimeoutExpired:
#         logging.error(f"Query in {query_filename} timed out.")
#         with open(output_file, 'w') as f:
#             f.write("time failed")
#     except subprocess.CalledProcessError as e:
#         logging.error(f"Query execution failed with an error: {str(e)}")
#         with open(output_file, 'w') as f:
#             f.write("execution failed")
#     else:
#         logging.info(f"Query in {query_filename} executed successfully.")

# def process_batch(batch_index, query_list, batch_size):
#     logging.info("Processing batch %d", batch_index)
#     start_index = batch_index * batch_size
#     end_index = start_index + batch_size
#     current_batch = query_list[start_index:end_index]
#     query_filename = f"./input_batch/batch{batch_index}.txt"
#     output_file = f'./aggregates/results{batch_index}.csv'
    
#     write_queries_to_file(current_batch, query_filename)
#     execute_query_script(query_filename, output_file)

# def multi_batch_processing(query_list, batch_size=10, max_workers=5):
#     total_queries = len(query_list)
#     num_batches = total_queries // batch_size

#     # Make sure the script is executable just once
#     subprocess.run(['chmod', '+x', 'test_query_run.sh'])
#     logging.basicConfig(level=logging.INFO)

#     # Use ThreadPoolExecutor to process batches in parallel
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         args = [(i, query_list, batch_size) for i in range(num_batches)]
#         executor.map(lambda p: process_batch(*p), args)

def create_batch(query_list, batch_size):
    """Split query_list into batches of queries in batch_size. """
    total_queries = len(query_list)
    num_batches = total_queries // batch_size
    for batch_index in range(num_batches):
        logging.info("Processing batch %d", batch_index)
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        current_batch = query_list[start_index:end_index]
        query_filename = f"./input_batch/batch{batch_index}.txt"
        # output_file = f'./aggregates/results{batch_index}.csv'
        write_queries_to_file(current_batch, query_filename)

# #STEP 2
# def batch_processing(query_files, connection_params, max_workers=5):
#     logging.basicConfig(level=logging.INFO)
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         all_results = {}
#         for file in query_files:
#             logging.info(f"Processing file {file}")
#             results = process_queries(file, connection_params)
#             all_results[file] = results
#     return all_results

########################################
#TESTING WITH MULTIPROCESSING
def execute_script(batch_file):
    """Executes the shell script on the given batch file."""
    subprocess.run(['./test_query_run.sh', batch_file])

def process_batch(batch_file):
    """
    Executes the script and aggregates text files, then cleans up the output directory.
    """
    source_folder = './outputs'
    try:
        clean_directory(source_folder)
        print("running bash file")
        subprocess.run(['./test_query_run.sh', batch_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute script for {batch_file}: {str(e)}")
    # Aggregate text files from the outputs folder
    # source_folder = './outputs'
    output_file = f'./aggregates/results{batch_file}.csv'
    aggregate_text_files(source_folder, output_file)
    print(f"Aggregated to {output_file}")
    # Clean up the outputs folder
    clean_directory(source_folder)

def clean_directory(directory):
    """
    Deletes all files in the specified directory to prepare for the next batch.
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {str(e)}')

def get_indices(csv_dir):
    # csv_dir_to_examine = 'merged_output.csv'
    merged = pd.read_csv(csv_dir)
    mask = (merged['results_returned'] != 'no_result') & (merged['results_returned'] != 'failed')
    # mask = merged['results_returned'] != 'no_result' or 'failed'
    failed_mask = merged['results_returned'] == 'failed'
    successful_indices = merged.index[mask].tolist()
    failed_indices = merged.index[failed_mask].tolist()
    return successful_indices, failed_indices


########################Previously working##############################################
def execute_query(query):
    # Create thread-local storage for database connections
    if not hasattr(execute_query, "conn"):
        execute_query.conn = mgclient.connect(host='127.0.0.1', port=7687)
        execute_query.conn.autocommit = True
    try:
        cursor = execute_query.conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    except Exception as e:
        print(f"Failed to execute query {query}: {str(e)}")
        return None
    finally:
        cursor.close()
# def execute_query(query):
#     # Create thread-local storage for database connections
#     if not hasattr(execute_query, "conn"):
#         execute_query.conn = mgclient.connect(host='127.0.0.1', port=7687)
#         execute_query.conn.autocommit = True
#     try:
#         cursor = execute_query.conn.cursor()
#         cursor.execute(query)
#         results = cursor.fetchall()
#         if not results:
#             # No results found, return a specific value that indicates empty results
#             return 'NoResult'
#         else:
#             return results
#     except Exception as e:
#         # print(f"Failed to execute query {query}: {str(e)}")
#         return None
#     finally:
#         cursor.close()
def run_queries_multithreaded(query_list, max_workers=5):
    # Create a ThreadPoolExecutor to manage multiple threads
    with threading.Thread.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map futures to their corresponding query indices
        future_to_index = {executor.submit(execute_query, query): i for i, query in enumerate(query_list)}

        # Create a list to store the indices of successful queries
        successful_indices = []
        failed_indices = []

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                if result:
                    print(f"Query at index {index} returned results.")
                    successful_indices.append(index)
                else:
                    print(f"Query at index {index} returned no results.")
            except Exception as e:
                print(f"Query execution failed for index {index}: {str(e)}")
                failed_indices.append(index)

        return successful_indices, failed_indices