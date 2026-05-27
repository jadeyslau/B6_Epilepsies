import pandas as pd
from mpi4py import MPI
import os
import glob
import b6_epilepsies as b6  # Ensure this is the correct module import

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define input variables
date = "251111"
box1 = "14"
box2 = "15"
exp = "PNPO_paradigm"
export = False  # Adjust if needed

name = date + "_" + box1 + "_" + box2 + "_" + exp
path = name + "/"

# Define data folder and output file (Only Final Processed Output)
processed_output_file = path + name + "_processed_df.csv"
# Define data folder and output file
data_folder = path + name + "_rawoutput/raw_converted_csv/"
# output_file = path + name + "_processed_df.csv"

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

print(f"Process {rank} handling {len(assigned_files)} files.")

# Define columns
cols = ['abstime', 'time', 'type', 'location', 'data1']

# Process assigned files
partial_dataframes = []
for file in assigned_files:
    print(f"Process {rank} reading {file}")

    df = pd.read_csv(file, usecols=cols, parse_dates=['abstime'], date_format="%Y-%m-%d %H:%M:%S")

    # **Filter rows where 'type' == 101**
    df = df[df['type'] == 101]

    if not df.empty:  # Avoid appending empty DataFrames
        partial_dataframes.append(df)

# Convert each process's list of DataFrames into a single DataFrame
local_df = pd.concat(partial_dataframes, ignore_index=True) if partial_dataframes else pd.DataFrame()

# Send Data in Chunks
CHUNK_SIZE = 500000  # Adjust based on memory capacity

if rank == 0:
    # **Master process collects data from all ranks (WITHOUT SAVING RAW CSV)**
    all_dataframes = [local_df] if not local_df.empty else []

    for i in range(1, size):
        print(f"Rank 0 waiting to receive from rank {i}")
        while True:
            received_chunk = comm.recv(source=i)

            # **Check for termination signal**
            if isinstance(received_chunk, str) and received_chunk == "END":
                break  # Stop receiving from this process

            all_dataframes.append(received_chunk)

    # **Merge all received data**
    final_df = pd.concat(all_dataframes, ignore_index=True) if all_dataframes else pd.DataFrame()

    # **Instantiate RawData Object (Pass Final Data Directly)**
    print("Initializing RawData object and processing data...")
    obj_raw = b6.RawData(date, box1, box2, exp)  # Create the RawData instance

    # **Pass Final DataFrame Directly to Processing (No Need to Save Raw CSV)**
    processed_df = obj_raw.prepare_raw_data(final_df)

    # **Save the final processed dataset as CSV**
    processed_df.to_csv(processed_output_file, index=False)

    print(f"Final processed dataset saved as {processed_output_file}")

else:
    # Other processes send their DataFrame in chunks
    print(f"Process {rank} sending data to rank 0 in chunks")
    for start in range(0, len(local_df), CHUNK_SIZE):
        chunk = local_df.iloc[start : start + CHUNK_SIZE]
        comm.send(chunk, dest=0)

    # Send termination signal
    comm.send("END", dest=0)  # Ensures rank 0 knows when to stop receiving

# Ensure all processes finish before exiting
comm.Barrier()
