import shutil
import pandas as pd
import os
import csv
import subprocess
import logging
import re
# import mgclient
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from multiprocessing import Manager
import shutil
import os
import copy

# lock = threading.Lock()
# counter = 0


def clear_folder(directory_path):
    # Remove the entire directory
    shutil.rmtree(directory_path)
    # Recreate the empty directory
    os.makedirs(directory_path)
# conn = mgclient.connect(host='127.0.0.1', port=7687)
# cursor = conn.cursor()

def write_queries_to_file(queries, filename):
    """Writes a list of queries to a specified file."""
    with open(filename, 'w') as file:
        for query in queries:
            if query == None:
                file.write(None)
            else:
                file.write(query + '\n')

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def y_aggregate_text_files(folder_path, output_file):
    # Initialize a list to store the aggregated data
    aggregated_data = []

    # List all CSV files in the specified folder
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Sort files based on their numerical names
    # file_list.sort(key=lambda f: int(re.search(r'\d+', f).group()))
    file_list.sort(key=natural_sort_key)

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
        # Join the rest of the lines for 'results_returned'
        if len(lines) > 1:
            results_returned = '; '.join([line.strip().replace('\n', ';') for line in lines[1:]])
        else:
            results_returned = copy.deepcopy(variables_returned)
        
        aggregated_data.append({
            "file_name": file_name,  # Add file name for reference
            "variables_returned": variables_returned,
            "results_returned": results_returned
        })

    # Create a DataFrame from the aggregated data
    result_df = pd.DataFrame(aggregated_data)
    
    # Sort the DataFrame by file name (which is now sorted numerically)
    result_df = result_df.sort_values('file_name', key=lambda x: x.map(natural_sort_key))
    
    # Remove the file_name column if you don't want it in the final output
    result_df = result_df.drop('file_name', axis=1)
    
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
    return merged_df

###############################
def x_execute_script(batch_index, batch_file):
    """Executes the shell script on the given batch file."""
    subprocess.run(['./query_batch_timeout.sh', batch_file, str(batch_index)])

def x_process_batch(batch_index, query_list, batch_size):
    global counter
    try:
        logging.info("Processing batch %d", batch_index)
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        current_batch = query_list[start_index:end_index]
        query_filename = f"./input_batch/batch{batch_index}.txt"
        
        write_queries_to_file(current_batch, query_filename)
        # x_execute_query_script(query_filename, output_file)
        x_execute_script(batch_index, query_filename)
        
        # Aggregate results
        folder_path = f'./outputs/{batch_index}'
        output_file = f'./aggregates/results{batch_index}.csv'
        y_aggregate_text_files(folder_path, output_file)

    except Exception as e:
        logging.error("Failed to process batch %d: %s", batch_index, str(e))

def x_multi_batch_processing(query_list, batch_size=10, max_workers=4):
    total_queries = len(query_list)
    num_batches = total_queries // batch_size

    # Make sure the script is executable just once
    subprocess.run(['chmod', '+x', 'query_batch_timeout.sh'])
    logging.basicConfig(level=logging.INFO)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(x_process_batch, range(num_batches), [query_list]*num_batches, [batch_size]*num_batches)


        
################################################################
# def create_batch(query_list, batch_size):
#     """Split query_list into batches of queries in batch_size. """
#     total_queries = len(query_list)
#     num_batches = total_queries // batch_size
#     for batch_index in range(num_batches):
#         logging.info("Processing batch %d", batch_index)
#         start_index = batch_index * batch_size
#         end_index = start_index + batch_size
#         current_batch = query_list[start_index:end_index]
#         query_filename = f"./input_batch/batch{batch_index}.txt"
#         # output_file = f'./aggregates/results{batch_index}.csv'
#         write_queries_to_file(current_batch, query_filename)

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
    merged = pd.read_csv(csv_dir)
    mask = (merged['results_returned'] != 'empty') & (merged['results_returned'] != 'failed') & (merged['results_returned'] != 'timeout fail')
    # mask = merged['results_returned'] != 'no_result' or 'failed'
    failed_mask = (merged['results_returned'] == 'failed')
    time_failed_mask = (merged['results_returned'] == 'timeout fail')
    successful_indices = merged.index[mask].tolist()
    failed_indices = merged.index[failed_mask].tolist()
    time_failed_indices = merged.index[time_failed_mask].tolist()
    return successful_indices, failed_indices, time_failed_indices


########################Previously working##############################################
# def execute_query(query):
#     # Create thread-local storage for database connections
#     if not hasattr(execute_query, "conn"):
#         execute_query.conn = mgclient.connect(host='127.0.0.1', port=7687)
#         execute_query.conn.autocommit = True
#     try:
#         cursor = execute_query.conn.cursor()
#         cursor.execute(query)
#         results = cursor.fetchall()
#         return results
#     except Exception as e:
#         print(f"Failed to execute query {query}: {str(e)}")
#         return None
#     finally:
#         cursor.close()
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
# def run_queries_multithreaded(query_list, max_workers=5):
#     # Create a ThreadPoolExecutor to manage multiple threads
#     with threading.Thread.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#         # Map futures to their corresponding query indices
#         future_to_index = {executor.submit(execute_query, query): i for i, query in enumerate(query_list)}

#         # Create a list to store the indices of successful queries
#         successful_indices = []
#         failed_indices = []

#         # Collect results as they complete
#         for future in concurrent.futures.as_completed(future_to_index):
#             index = future_to_index[future]
#             try:
#                 result = future.result()
#                 if result:
#                     print(f"Query at index {index} returned results.")
#                     successful_indices.append(index)
#                 else:
#                     print(f"Query at index {index} returned no results.")
#             except Exception as e:
#                 print(f"Query execution failed for index {index}: {str(e)}")
#                 failed_indices.append(index)

#         return successful_indices, failed_indices
    

# def currently_process_batches(batch_folder):
#     # Find all batch files
#     batch_files = [os.path.join(batch_folder, f) for f in os.listdir(batch_folder) if f.startswith('batch') and f.endswith('.txt')]
#     # Ensure the script is executable
#     subprocess.run(['chmod', '+x', 'test_query_run.sh'])
    
#     # Process each batch in parallel
#     with Pool(processes=os.cpu_count()) as pool:
#         pool.map(execute_script, batch_files)

#     # Aggregate results
#     counter = 0
#     folder_path = './outputs'
#     output_file = './aggregates/results{counter}.csv'
#     aggregate_text_files(folder_path, output_file)
#     counter += 1

#     # Merge CSV files if needed
#     merge_csv_files('aggregates', 'multiproc_merged_output.csv')
#     # Get indices of successful and failed queries
#     get_indices('multiproc_merged_output.csv')