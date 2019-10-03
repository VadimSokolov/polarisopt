import sys
import cplex_functions

if __name__ == "__main__":
    base_directory = sys.argv[1]

    # get time
    timestamp = cplex_functions.time()

    # setup directories, return all as one (just for fun)    
    source_directory, run_directory, input_directory, results_directory = cplex_functions.make_directories(base_directory, timestamp)

    # call additional scripts
    cplex_functions.run_external_scripts(timestamp, source_directory, input_directory, results_directory)
