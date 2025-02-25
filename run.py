import pandas as pd
from mpi4py import MPI
import os
import glob

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define input variables
date = "250121"
box1 = "14"
box2 = "15"
exp = "PLPBP"

name = date + "_" + box1 + "_" + box2 + "_" + exp
path = name + "/"

# Define data folder and output file
data_folder = path + name + "_rawoutput/raw_converted_csv/"
output_file = path + name + "_raw_df.csv"

# Get list of all CSV files
csv_files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))

# Divide files among MPI processes
files_per_proc = len(csv_files) // size
extra_files = len(csv_files) % size

if rank < extra_files:
    start = rank * (files_per_proc + 1)
    end = start + files_per_proc + 1
else:
    start = rank * files_per_proc + extra_files
    end = start + files_per_proc

assigned_files = csv_files[start:end]

print("Process", rank, "handling", len(assigned_files), "files.")

# Define columns
cols = ['abstime', 'time', 'channel', 'type', 'location', 'data1']

# Process assigned files
partial_dataframes = []
for file in assigned_files:
    print("Process", rank, "reading", file)
    df = pd.read_csv(file, usecols=cols, parse_dates=['abstime'], date_format="%Y-%m-%d %H:%M:%S")
    partial_dataframes.append(df)

# Convert each process's list of DataFrames into a single DataFrame
local_df = pd.concat(partial_dataframes, ignore_index=True) if partial_dataframes else pd.DataFrame()

# Send Data in Chunks
CHUNK_SIZE = 500000  # Adjust this based on your memory capacity

if rank == 0:
    # Master process initializes the CSV file and writes incoming data chunk-by-chunk
    with open(output_file, 'w') as f:
        if not local_df.empty:
            local_df.to_csv(f, index=False)
        for i in range(1, size):
            print("Rank 0 waiting to receive from rank", i)
            while True:
                received_chunk = comm.recv(source=i)

                # **Fix: Ensure we properly check for the termination signal**
                if isinstance(received_chunk, str) and received_chunk == "END":
                    break  # Stop receiving from this process

                received_chunk.to_csv(f, header=False, index=False)

    print("Final dataset saved as", output_file)

else:
    # Other processes send their DataFrame in chunks
    print("Process", rank, "sending data to rank 0 in chunks")
    for start in range(0, len(local_df), CHUNK_SIZE):
        chunk = local_df.iloc[start : start + CHUNK_SIZE]
        comm.send(chunk, dest=0)

    # Send termination signal
    comm.send("END", dest=0)  # Ensures rank 0 knows when to stop receiving

# Ensure all processes finish before exiting
comm.Barrier()
