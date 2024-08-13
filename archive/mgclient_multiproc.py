import mgclient
from multiprocessing import Pool, current_process
import os
from threading import Timer


# Function to establish a new database connection
def connect_db():
    return mgclient.connect(host='127.0.0.1', port=7687)

# Function to execute a single query
# def execute_query(query):
#     # Connect to the database
#     conn = connect_db()
#     try:
#         cursor = conn.cursor()
#         cursor.execute(query)
#         results = cursor.fetchall()
#         if not results:
#             # No results found, return a specific value that indicates empty results
#             return None
#         else:
#             # Return the query and its results
#             return (query, results)
#     except Exception as e:
#         print(f"Failed to execute query {query} on process {current_process().name}: {str(e)}")
#         return None
#     finally:
#         cursor.close()
#         conn.close()

def execute_query_withtimeout(query):
    conn = connect_db()
    try:
        cursor = conn.cursor()
        
        # Setup a timer to stop the query after a timeout
        def timeout_handler():
            conn.close()  # Closing connection should cancel the query
            print(f"Query timed out: {query}")
        
        timer = Timer(3, timeout_handler)  # Set timeout to 3 seconds
        timer.start()

        cursor.execute(query)
        results = cursor.fetchall()

        timer.cancel()  # Cancel the timer if the query completes in time

        if not results:
            # No results found, return a specific value that indicates empty results
            return None
        else:
            # Return the query and its results
            return (query, results)

    except Exception as e:
        print(f"Failed to execute query {query} on process {current_process().name}: {str(e)}")
        return None
    finally:
        cursor.close()
        conn.close()
        
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