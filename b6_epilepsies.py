# Standard library imports
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
import glob
import sys

# Third-party imports
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.stats import shapiro, normaltest, kruskal, mannwhitneyu, f_oneway, ttest_ind
from statannotations.Annotator import Annotator

from concurrent.futures import ThreadPoolExecutor
import dask.dataframe as dd
from dask.distributed import Client

import zipfile
import json
import xml.etree.ElementTree as ET
import pandas as pd

sns.set_theme(style="whitegrid")
plt.rcParams['savefig.transparent'] = True

class Experiment:
    def __init__(self, date, box1, box2, exp, export=False, nbox=2, cbox=None, omit=None):
        self.date = date
        self.box1 = box1
        self.box2 = box2
        self.exp  = exp
        self.omit = omit# if omit else []  # 'omit' available to all subclasses

        self.export = export

        # If cbox is not provided, default it to the value of box1
        self.cbox = cbox if cbox is not None else self.box1

        self.name = f"{self.date}_{self.box1}_{self.box2}_{self.exp}"
        self.path = f"{self.name}/"

        self.filename = f"{self.name}/{self.date}_{self.box1}_{self.box2}_{self.exp}.csv"

        # Correct experiment start timestamp
        self.correct_start_time = self.get_correct_start_time_from_middur_xls()


        # Define ZT0 (e.g., 9:00 AM on the start date)
        zt0_time = datetime.strptime("09:00:00", "%H:%M:%S").time()
        self.zt0 = datetime.combine(self.correct_start_time.date(), zt0_time)


        # genotype_file = f"{self.path}{self.date}_{self.cbox}_genotypeMap.xlsx"
        self.geno_map_files = self.find_map_files(self.path)['Genotype Map Files'][0]
        self.cond_map_files = self.find_map_files(self.path)['Condition Map Files'][0]


    def get_correct_start_time_from_middur_xls(self):

        # Columns to read from the CSV file
        cols = ["stdate", "sttime"]

        # Read and add box identifiers
        temp_df  = pd.read_csv(self.filename, usecols=cols, low_memory=True, header=0, nrows=2)
        # Combine 'stdate' and 'sttime' into a single datetime column
        temp_df['stdate_sttime'] = pd.to_datetime(temp_df['stdate'] + ' ' + temp_df['sttime'], dayfirst=False)
        print(f"Start of experiment: {temp_df['stdate_sttime'][0]}")

        return temp_df['stdate_sttime'][0]

    def get_genotype_df(self):
        # geno = self.find_map_files(self.path)
        # self.geno_map_files = self.find_map_files(self.path)['Genotype Map Files'][0]
        # self.cond_map_files = self.find_map_files(self.path)['Condition Map Files'][0]
        # Load genotype data
        genotype_file = next(iter(glob.glob(os.path.join(self.path, "*_genotypes.csv"))), None)
        # genotype_df = pd.read_csv(genotype_file) if genotype_file else self.get_genotype_df()
        # print(f"Using genotype file: {genotype_file}" if genotype_file else "No genotype file found. Generating from Excel file.")

        if genotype_file:
            genotype_df = pd.read_csv(genotype_file)
            print(f"Using genotype file: {genotype_file}" if genotype_file else "No genotype file found. Generating from Excel file.")
        else:

            df_list = []

            for i, geno_file in enumerate(self.geno_map_files):

                current_box = re.search(r'_(\d+)_genotypeMap', geno_file).group(1)


                # Tidy genotype and condMap data, adding box identifiers
                genotype_data = pd.read_excel(self.path+geno_file) # If the Excel file is open, this might throw an engine error
                geno_df = self.strip_96well_data(genotype_data, current_box, 'genotype')

                df_list.append(geno_df)

            genotype_df = pd.concat(df_list, ignore_index=True)

        genotype_df.box = genotype_df.box.astype(str)

        return genotype_df

    def get_condition_df(self):
        print('We have a condition!')
        cond_df_list = []

        for i, cond_file in enumerate(self.cond_map_files):
            box_name = f"box{i+1}"  # Dynamically create the attribute name
            current_box = getattr(self, box_name, None)  # Access the attribute

            # Load the condition data from the Excel file
            condition_data = pd.read_excel(self.path + cond_file)

            # Tidy genotype and condMap data, adding box identifiers
            cond_df = self.strip_96well_data(condition_data, current_box, 'condition')

            # Remove the unit from the condition column (keeping only numeric values)
            cond_df['condition'] = cond_df['condition'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)

            cond_df_list.append(cond_df)

        # Combine all condition DataFrames
        condition_df = pd.concat(cond_df_list, ignore_index=True)
        return condition_df

    def get_geno_cond_map(self, file, box, datatype='genotype'):
        data = pd.read_excel(file)
        df   = self.strip_96well_data(data, box, datatype)
        return df

    def strip_96well_data(self, data, box, datatype='genotype'): #extracts genotype/condition details from excel file
        plate_df = data.iloc[0:8, 0:13]
        plate_df['box'] = box
        plate_df = plate_df.rename(columns={'Unnamed: 0': 'row'}).melt(id_vars=['row', 'box'], var_name='Column', value_name=datatype)
        plate_df['well'] = plate_df['row'] + plate_df['Column'].astype(str)
        plate_df = plate_df[['box', 'well', datatype]] # Only keep pertinent data
        return plate_df

    # Function to calculate the 'CLOCK' column
    def calculate_clock(self, series):
        def compute_clock(start_datetime):
            reference_time = start_datetime.replace(hour=9, minute=0, second=0)
            if start_datetime < reference_time:
                reference_time -= timedelta(days=1)
            return (start_datetime - reference_time).total_seconds() / 3600

        return pd.concat([pd.Series([""]), series.apply(compute_clock)])

    def get_well_label(self, well_number):
        # Determine plate and adjust well number for row-column calculation
        plate_offset = (well_number - 1) // 96 * 96
        adjusted_well_number = well_number - plate_offset

        # Row (A-H) and column (1-12) calculation
        row = chr(65 + ((adjusted_well_number - 1) % 8))  # 65 is ASCII for 'A'
        col = ((adjusted_well_number - 1) // 8) + 1
        return f"{row}{col}"

    def find_map_files(self, folder_path):
        """
        Search for files ending in '_genotypeMap.xlsx' and '_condMap.xlsx' in the specified folder.

        Args:
            folder_path (str): Path to the folder to search.

        Returns:
            dict: A dictionary containing lists of genotype and condition map files.
        """
        # Initialize lists to hold filenames
        genotype_map_files = []
        cond_map_files = []

        # Iterate through the folder
        for file_name in os.listdir(folder_path):
            # Check for files ending in "_genotypeMap.xlsx"
            if file_name.endswith("_genotypeMap.xlsx"):
                genotype_map_files.append(file_name)
            # Check for files ending in "_condMap.xlsx"
            elif file_name.endswith("_condMap.xlsx"):
                cond_map_files.append(file_name)

        return {
            "Genotype Map Files" : [genotype_map_files, len(genotype_map_files)],
            "Condition Map Files": [cond_map_files,len(cond_map_files)]}

class RawData(Experiment):
    def __init__(self, date, box1, box2, exp, export=False, nbox=2, cbox=None, omit=None):
        print('Initialising RawData...')
        # Call the parent class's __init__
        super().__init__(date, box1, box2, exp, export, omit=omit)

        processed_file = f"{self.path}{self.name}_processed.csv"
        raw_file       = f"{self.path}{self.name}_raw_df.csv"

        if not os.path.exists(raw_file):
            print('_raw_df.csv not found. Need to combine.')
            raw_df  = self.combine_csv_files_dask()
            print('CSV Files combined')
            self.df = self.prepare_raw_data(raw_file)
            # self.df = pd.read_csv(processed_file)
            print('Complete.')
        elif not os.path.exists(processed_file):
            self.df = self.prepare_raw_data(raw_file)
        else:
            print(f'Instantiated using {processed_file}, no need to prepare raw data.')
            self.df = pd.read_csv(processed_file)
            print('Done.')

        # mega_df_file = f"{self.path}{self.name}_raw_df.csv"
        # output_file  = f"{self.path}{self.name}_processed.csv"
        # # mega_df_file_parquet = f"{self.path}{self.name}_raw_df.parquet"
        #
        # cols         = ["abstime", "time", "type", "location", "data1"]
        #
        # if input_file:
        #     mega_df_file = input_file
        # else:
        #     if not os.path.exists(mega_df_file):
        #         self.combine_csv_files_dask()
        #         # self.combine_csv_files()
        #         print("The file does not exist. CSV files need to be combined. Exiting program.")
        #         sys.exit()
        #
        #     print(f"Preparing data from {mega_df_file}.")
        #     dirty_data = pd.read_csv(mega_df_file, usecols=cols)


    # def prepare_raw_data(self, input_file=None):
    #     # mega_df_file = f"{self.path}{self.name}_raw_df.csv"
    #     output_file  = f"{self.path}{self.name}_processed.csv"
    #     # # mega_df_file_parquet = f"{self.path}{self.name}_raw_df.parquet"
    #     #
    #     cols         = ["abstime", "time", "type", "location", "data1"]
    #     #
    #     # if input_file:
    #     #     mega_df_file = input_file
    #     # else:
    #     #     if not os.path.exists(mega_df_file):
    #     #         self.combine_csv_files_dask()
    #     #         # self.combine_csv_files()
    #     #         print("The file does not exist. CSV files need to be combined. Exiting program.")
    #     #         sys.exit()
    #     #
    #     #     print(f"Preparing data from {mega_df_file}.")
    #     #     dirty_data = pd.read_csv(mega_df_file, usecols=cols)
    #
    #     # dirty_data = pd.read_parquet(mega_df_file_parquet, columns=cols)
    #
    #     dirty_data = pd.read_csv(input_file, usecols=cols)
    #
    #
    #     # Filter rows and create a copy
    #     filtered_df = dirty_data[dirty_data['type'] == 101].copy()
    #
    #     # ADJUST TIME
    #     # Vectorized operations for creating new columns
    #     time_in_seconds = filtered_df["time"] / 1_000_000
    #     filtered_df.loc[:, "fullts"] = self.correct_start_time + pd.to_timedelta(time_in_seconds, unit="s")
    #     filtered_df.loc[:, "zhrs"] = (filtered_df["fullts"] - pd.Timestamp(self.zt0)).dt.total_seconds() / 3600
    #     filtered_df.loc[:, "exsecs"] = time_in_seconds
    #
    #
    #     # Move the new columns to the front
    #     # columns_order = ["fullts", "zhrs", "exsecs"] + [col for col in filtered_df.columns if col not in ["fullts", "zhrs", "exsecs"]]
    #     # filtered_df = filtered_df[columns_order]
    #
    #     genotype_df = self.get_genotype_df()
    #
    #     # Adds columns box. well, and plate
    #     filtered_df = self.convert_location_column(filtered_df)
    #
    #     if self.cond_map_files:
    #         condition_df = self.get_condition_df()
    #         # Merge data on Location and Box
    #         merged_data = filtered_df.merge(genotype_df, on=['box', 'well'], how='left').merge(condition_df, on=['box', 'well'], how='left')
    #     else:
    #         merged_data = filtered_df.merge(genotype_df, on=['box', 'well'], how='left')
    #
    #
    #     merged_data['genotype'].dropna(inplace=True)
    #     merged_data = merged_data[~merged_data['genotype'].isin(['empty', 'NaN', np.nan])]
    #
    #     # Ensure elapsed_time is calculated
    #     reference_time = merged_data['fullts'].min()
    #     merged_data['elapsed_time'] = (merged_data['fullts'] - reference_time).dt.total_seconds() / 60
    #
    #     print('Dropping unecessary columns and duplicates...')
    #     final_df = merged_data.drop(columns=['zhrs', 'exsecs','abstime', 'time','type','plate','location']).drop_duplicates(subset=['fullts','well','box'])
    #
    #     final_df_cols = ["fullts", "elapsed_time", "box", "well", "genotype", "data1"]
    #
    #     columns_order = final_df_cols + [col for col in final_df.columns if col not in final_df_cols]
    #     final_df = final_df[columns_order]
    #     final_df.astype({'box':'int64'})
    #
    #     final_df.to_csv(output_file, index=False)
    #     print(f"Saved prepped df to {output_file}")
    #
    #     print('Done')
    #     return final_df

    # def prepare_raw_data(self, input_file=None):
    #     output_file = f"{self.path}{self.name}_processed.csv"
    #     cols = ["abstime", "time", "type", "location", "data1"]
    #
    #     # Determine source file
    #     if input_file is None:
    #         parquet_file = f"{self.path}{self.name}_raw_df.parquet"
    #         csv_file = f"{self.path}{self.name}_raw_df.csv"
    #
    #         if not os.path.exists(parquet_file):
    #             if os.path.exists(csv_file):
    #                 print("Parquet not found — converting from CSV (one-time step)...")
    #                 self.combine_csv_to_parquet()
    #             else:
    #                 print("No raw data found. Running combine step first...")
    #                 self.combine_csv_files_dask()
    #                 self.combine_csv_to_parquet()
    #
    #         print(f"Reading from {parquet_file}")
    #         dirty_data = pd.read_parquet(parquet_file, columns=cols)
    #     else:
    #         # Allow CSV or parquet input
    #         if input_file.endswith('.parquet'):
    #             dirty_data = pd.read_parquet(input_file, columns=cols)
    #         else:
    #             dirty_data = pd.read_csv(input_file, usecols=cols)
    #
    #     # Filter early
    #     filtered_df = dirty_data[dirty_data['type'] == 101].copy()
    #     del dirty_data  # free memory immediately
    #
    #     # Time columns
    #     time_in_seconds = filtered_df["time"] / 1_000_000
    #     filtered_df["fullts"] = self.correct_start_time + pd.to_timedelta(time_in_seconds, unit="s")
    #
    #     # Merge genotype + conditions
    #     filtered_df = self.convert_location_column(filtered_df)
    #     genotype_df = self.get_genotype_df()
    #
    #     if self.cond_map_files:
    #         condition_df = self.get_condition_df()
    #         merged_data = (filtered_df
    #                        .merge(genotype_df, on=['box', 'well'], how='left')
    #                        .merge(condition_df, on=['box', 'well'], how='left'))
    #     else:
    #         merged_data = filtered_df.merge(genotype_df, on=['box', 'well'], how='left')
    #
    #     del filtered_df
    #
    #     # Clean genotypes
    #     merged_data = merged_data[
    #         merged_data['genotype'].notna() &
    #         ~merged_data['genotype'].isin(['empty', 'NaN'])
    #     ]
    #
    #     # Elapsed time
    #     reference_time = merged_data['fullts'].min()
    #     merged_data['elapsed_time'] = (merged_data['fullts'] - reference_time).dt.total_seconds() / 60
    #
    #     # Tidy up
    #     drop_cols = ['abstime', 'time', 'type', 'plate', 'location']
    #     drop_cols = [c for c in drop_cols if c in merged_data.columns]
    #     final_df = (merged_data
    #                 .drop(columns=drop_cols)
    #                 .drop_duplicates(subset=['fullts', 'well', 'box'])
    #                 .astype({'box': 'int64'}))
    #
    #     final_df_cols = ["fullts", "elapsed_time", "box", "well", "genotype", "data1"]
    #     columns_order = final_df_cols + [c for c in final_df.columns if c not in final_df_cols]
    #     final_df = final_df[columns_order]
    #
    #     # Still saves as CSV for your downstream steps
    #     final_df.to_csv(output_file, index=False)
    #     print(f"Saved prepped df to {output_file}")
    #     return final_df

    def prepare_raw_data(self, input_file=None, chunk_size=500_000):
        output_file = f"{self.path}{self.name}_processed.csv"
        cols = ["abstime", "time", "type", "location", "data1"]

        if input_file is None:
            input_file = f"{self.path}{self.name}_raw_df.csv"

        genotype_df = self.get_genotype_df()
        condition_df = self.get_condition_df() if self.cond_map_files else None

        print(f"Processing {input_file} in chunks of {chunk_size}...")

        # --- Pass 1: filter, merge, write to temp file ---
        temp_file = f"{self.path}{self.name}_temp.csv"
        first_chunk = True
        total_rows = 0

        for i, chunk in enumerate(pd.read_csv(input_file, usecols=cols, chunksize=chunk_size)):
            chunk = chunk[chunk['type'] == 101].copy()

            if chunk.empty:
                print(f"  Chunk {i+1}: no type==101 rows, skipping")
                continue

            time_in_seconds = chunk["time"] / 1_000_000
            chunk["fullts"] = self.correct_start_time + pd.to_timedelta(time_in_seconds, unit="s")

            chunk = self.convert_location_column(chunk)
            chunk = chunk.merge(genotype_df, on=['box', 'well'], how='left')
            if condition_df is not None:
                chunk = chunk.merge(condition_df, on=['box', 'well'], how='left')

            chunk = chunk[
                chunk['genotype'].notna() &
                ~chunk['genotype'].isin(['empty', 'NaN'])
            ]

            drop_cols = [c for c in ['abstime', 'time', 'type', 'plate', 'location']
                         if c in chunk.columns]
            chunk = chunk.drop(columns=drop_cols)

            chunk.to_csv(temp_file, mode='w' if first_chunk else 'a',
                         header=first_chunk, index=False)
            total_rows += len(chunk)
            first_chunk = False
            print(f"  Chunk {i+1}: kept {len(chunk)} rows (total so far: {total_rows})")

            del chunk
            import gc; gc.collect()

        if first_chunk:
            print("No valid data found!")
            return pd.DataFrame()

        # --- Pass 2: read the much smaller temp file for elapsed_time + dedup ---
        print(f"Pass 2: reading {total_rows} processed rows for elapsed_time + dedup...")
        final_df = pd.read_csv(temp_file, parse_dates=['fullts'])

        reference_time = final_df['fullts'].min()
        final_df['elapsed_time'] = (final_df['fullts'] - reference_time).dt.total_seconds() / 60
        final_df = final_df.drop_duplicates(subset=['fullts', 'well', 'box'])
        final_df = final_df.astype({'box': 'int64'})

        final_df_cols = ["fullts", "elapsed_time", "box", "well", "genotype", "data1"]
        columns_order = final_df_cols + [c for c in final_df.columns if c not in final_df_cols]
        final_df = final_df[columns_order]

        final_df.to_csv(output_file, index=False)
        os.remove(temp_file)  # clean up

        print(f"Saved to {output_file} — shape: {final_df.shape}")
        return final_df

    def combine_csv_to_parquet(self, chunk_size=500_000):
        """
        Convert the mega CSV to parquet in chunks, keeping your existing
        CSV combine step but producing a much faster-to-read output.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        csv_file = f"{self.path}{self.name}_raw_df.csv"
        parquet_file = f"{self.path}{self.name}_raw_df.parquet"
        cols = ["abstime", "time", "type", "location", "data1"]

        if not os.path.exists(csv_file):
            self.combine_csv_files_dask()

        print(f"Converting {csv_file} to parquet in chunks...")
        writer = None

        for i, chunk in enumerate(pd.read_csv(csv_file, usecols=cols, chunksize=chunk_size)):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(parquet_file, table.schema, compression='snappy')
            writer.write_table(table)
            print(f"  Written chunk {i + 1} ({len(chunk)} rows)")

        if writer:
            writer.close()

        print(f"Saved parquet to {parquet_file}")
        return parquet_file

    def convert_location_column(self, df, location_column="location"):
        """
        Efficiently converts a 'location' column in a DataFrame into 'plate', 'well', and 'box' columns for large datasets.

        Args:
        df (pd.DataFrame): Input DataFrame containing a 'location' column.
        location_column (str): Name of the location column (default: 'location').

        Returns:
        pd.DataFrame: DataFrame with additional 'plate', 'well', and 'box' columns.
        """
        # Extract numeric part of the location
        numeric_locations = df[location_column].str.extract(r'(\d+)').astype(int)[0]

        # Calculate plate number
        plates = (numeric_locations - 1) // 96 + 1

        # Calculate position within plate
        well_numbers_within_plate = (numeric_locations - 1) % 96

        # Calculate row (A-H) and column (1-12)
        rows = (well_numbers_within_plate // 12).map(lambda x: chr(65 + x))  # Convert to letter
        columns = (well_numbers_within_plate % 12 + 1)

        # Combine row and column to create well identifiers
        wells = rows + columns.astype(str)

        # Assign box based on plate
        boxes = plates.map({1: self.box1, 2: self.box2})

        # Add new columns to the DataFrame
        df["plate"] = plates
        df["well"] = wells
        df["box"] = boxes

        return df

    # def transform_time(self, row):
    #     # Convert `time` from microseconds to seconds
    #     time_in_seconds = row["time"] / 1_000_000
    #     # Calculate full timestamp (`fullts`)
    #     full_timestamp = self.correct_start_time + timedelta(seconds=time_in_seconds)
    #     # Calculate Zeitgeber hours (`zhrs`)
    #     zhrs = (full_timestamp - self.zt0).total_seconds() / 3600
    #     return pd.Series([full_timestamp, zhrs, time_in_seconds])

    def combine_csv_MPI(self, csv_files):
    # def combine_csv_files(self, input_folder, output_file=None):
        """
        Combines all CSV files in a folder into a single DataFrame.
        You first need to use the bash script (found in Scripts) in Terminal to batch convert xls to csv with SSCONVERT.
        e.g. ./batch_convert_xls_to_csv.sh 241107_16_17_PNPO_PTZ/241107_16_17_PNPO_PTZ_rawoutput
        This method will not work if there is an existing combined csv inside the input folder (csv_folder).

        Args:
            input_folder (str): Path to the folder containing CSV files.
            output_file (str, optional): Path to save the combined DataFrame as a CSV.

        Returns:
            pandas.DataFrame: Combined DataFrame of all CSV files.
        """
        csv_folder  = f"{self.path}{self.name}_rawoutput/raw_converted_csv/"
        output_file = f"{self.path}{self.name}_raw_df.csv"

        # List and sort all CSV files in the input folder by numeric order
        csv_files = sorted(csv_files)

        cols = ['abstime', 'time', 'channel', 'type', 'location', 'data1']

        # Initialize an empty list to store DataFrames
        dataframes = []

        # Iterate through each CSV file
        for csv_file in csv_files:
            file_path = os.path.join(csv_folder, csv_file)
            print(f"Reading {file_path}")
            df = pd.read_csv(file_path, usecols=cols, parse_dates=['abstime'])  # Adjust for delimiter if necessary
            dataframes.append(df)

        # Combine all DataFrames into one
        partial_df = pd.concat(dataframes, ignore_index=True)
        # print("All files have been combined into a single DataFrame.")

        # Save the combined DataFrame as a CSV if specified
        # if output_file:
        #     self.mega_dataframe.to_csv(output_file, index=False)
        #     print(f"Combined DataFrame saved to {output_file}")

        return partial_df

    def combine_csv_files(self, output_file=None):
    # def combine_csv_files(self, input_folder, output_file=None):
        """
        Combines all CSV files in a folder into a single DataFrame.
        You first need to use the bash script (found in Scripts) in Terminal to batch convert xls to csv with SSCONVERT.
        e.g. ./batch_convert_xls_to_csv.sh 241107_16_17_PNPO_PTZ/241107_16_17_PNPO_PTZ_rawoutput
        This method will not work if there is an existing combined csv inside the input folder (csv_folder).

        Args:
            input_folder (str): Path to the folder containing CSV files.
            output_file (str, optional): Path to save the combined DataFrame as a CSV.

        Returns:
            pandas.DataFrame: Combined DataFrame of all CSV files.
        """
        csv_folder  = f"{self.path}{self.name}_rawoutput/raw_converted_csv/"
        output_file = f"{self.path}{self.name}_raw_df.csv"

        # List and sort all CSV files in the input folder by numeric order
        csv_files = sorted(
            [f for f in os.listdir(csv_folder) if f.endswith('.csv')],
            key=lambda x: int(re.search(r'_(\d+)\.csv$', x).group(1))  # Match digits at the end
        )

        if not csv_files:
            print(f"No CSV files found in {csv_folder}.")
            return None

        cols = ['abstime', 'time', 'channel', 'type', 'location', 'data1']

        # Initialize an empty list to store DataFrames
        dataframes = []

        # Iterate through each CSV file
        for csv_file in csv_files:
            file_path = os.path.join(csv_folder, csv_file)
            print(f"Reading {file_path}")
            df = pd.read_csv(file_path, usecols=cols, parse_dates=['abstime'])  # Adjust for delimiter if necessary
            dataframes.append(df)

        # Combine all DataFrames into one
        self.mega_dataframe = pd.concat(dataframes, ignore_index=True)
        print("All files have been combined into a single DataFrame.")

        # Save the combined DataFrame as a CSV if specified
        if output_file:
            self.mega_dataframe.to_csv(output_file, index=False)
            print(f"Combined DataFrame saved to {output_file}")

        return self.mega_dataframe

    def combine_csv_files_dask(self, output_file=None):
        # Define input and output paths
        csv_folder = f"{self.path}{self.name}_rawoutput/raw_converted_csv/"
        output_file = output_file or f"{self.path}{self.name}_raw_df.csv"

        # Get the list of CSV files
        csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
        total_files = len(csv_files)

        if total_files == 0:
            print("No CSV files found to process.")
            return None

        print(f"Found {total_files} CSV files. Starting the combination process...")

        # Define consistent dtypes to avoid type conflicts
        dtype = {
            'abstime': 'object',  # Will be parsed later as datetime
            'time': 'object',
            'channel': 'float64',
            'type': 'float64',
            'location': 'object',
            'data1': 'float64'
        }

        # Use a pattern to load all CSV files
        csv_files_pattern = os.path.join(csv_folder, "*.csv")

        # **Limit memory per partition using `blocksize="100MB"`**
        ddf = dd.read_csv(
            csv_files_pattern,
            usecols=['abstime', 'time', 'channel', 'type', 'location', 'data1'],
            dtype=dtype,
            blocksize="100MB"  # Prevent loading too much into memory
        )

        # Convert 'abstime' column safely (Handle both numeric & datetime formats)
        ddf['abstime'] = dd.to_datetime(
            dd.to_numeric(ddf['abstime'], errors='coerce'),
            unit='ms', errors='coerce'
        ).fillna(
            dd.to_datetime(ddf['abstime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        )

        # Drop rows with invalid abstime
        ddf = ddf[ddf['abstime'].notnull()]

        # Save output
        if output_file:
            print(f"Writing combined data to {output_file}...")
            ddf.to_csv(output_file, index=False, single_file=True)  # Save efficiently

        print(f"Combination completed. Data saved to {output_file}.")

        return ddf

    def plot_fish_count_hist(self, box_well_list, full_df=None, col='data1', threshold=40):
        """
        I AM NOT SURE THIS METHOD IS USEFUL ANYMORE, MAY DELETE LATER. JL
        THIS IS THE FIRST V RUDIMENTARY COUNT OF 'EVENTS' WHICH ONLY COUNTS FRAME ABOVE THRESHOLD

        Plots a histogram of frame counts for single or multiple fish.

        - If given one fish, it plots a single histogram.
        - If given multiple fish, it arranges them in a subplot grid dynamically.

        Usage:
        all_hets = df_raw[df_raw["genotype"] == "HET"]
        all_hets_grouped = list(all_hets.groupby(["box", "well"]))

        # Extract (box, well) tuples
        fish_list = [key for key, _ in all_hets_grouped]

        # Pass all fish at once
        ald_lys_obj_raw.plot_fish_count_hist(box_well_list=fish_list)

        OR
        ald_lys_obj_raw.plot_fish_count_hist(box_well_list=fish_list, full_df=df_smoothed, col='smoothed_data1')

        """
        if full_df is None:
            full_df = self.df

        # Ensure input is a list of (box, well) tuples
        if isinstance(box_well_list[0], (int, str)):
            box_well_list = [box_well_list]

        num_fish = len(box_well_list)

        # Decide subplot layout (2 or 3 columns depending on odd/even count)
        # cols = 2 if num_fish % 2 == 0 else 3
        cols = 5
        rows = int(np.ceil(num_fish / cols))  # Compute number of rows dynamically

        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharex=True)
        axes = np.array(axes).flatten()  # Flatten for easy iteration

        for i, box_well in enumerate(box_well_list):
            box, well = box_well

            # Filter data
            fish_df = full_df[(full_df['box'] == box) & (full_df['well'] == well)]
            df = fish_df.sort_values(by="fullts")

            # print(box,type(int(box)), well, fish_df)
            # Boolean column indicating if data1 is above the threshold
            df["above_threshold"] = df[col] >= threshold

            # Compute frame count durations
            frame_count = 0
            count_durations = []
            above_threshold = False

            for row in df.itertuples(index=False):
                if row.above_threshold:
                    if not above_threshold:
                        frame_count = 1
                        above_threshold = True
                    else:
                        frame_count += 1
                else:
                    if above_threshold:
                        count_durations.append(frame_count)
                        above_threshold = False
                        frame_count = 0

            if above_threshold:
                count_durations.append(frame_count)

            # print(fish_df['genotype'])
            # Plot histogram
            sns.histplot(count_durations, kde=True, edgecolor="black", alpha=0.7, ax=axes[i])
            axes[i].set_yscale("log")
            axes[i].set_xlabel("Frames")
            axes[i].set_ylabel("log(Count)")
            axes[i].set_title(f"Box: {box}, Well: {well}; {fish_df['genotype'].unique()[0]}")

        # Remove unused subplots
        # for j in range(i + 1, len(axes)):
        #     fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def plot_single_fish_activity(self, well, event_windows, title_prefix="", drug_app=None, filterY=None, hue='genotype', figsize=(20, 18)):
        """
        Plots fish activity from a given dataframe with event highlights.

        Parameters:
        - df: pandas DataFrame containing the data
        - well: str, well identifier to filter the data
        - event_windows: list of tuples, each tuple contains (start_time, end_time, title)
        - title_prefix: str, prefix for subplot titles
        - figsize: tuple, figure size

        Usage:
        pnpo_ptz_obj_raw.plot_single_fish_activity(well='H2', event_windows=[
            (13, 13.5, "Seizure Event, Example 1"),
            (22.8, 23.3, "Seizure Event, Example 2"),
            (29.7, 30.2, "Seizure Event, Example 3"),
            (38, 38.5, "Normal Swimming Period, Example 4")
        ], drug_app=[5, 11.37], filterY=150, hue='condition')
        """


        df = self.df

        # Filter data for the selected well
        if filterY:
            fish_data = df[(df['well'] == well) & (df['data1'] < filterY)]
        else:
            fish_data = df[df['well'] == well]

        if 'condition' in fish_data.columns:
            fish_data['condition'] = fish_data['condition'].astype(int)
            condition = fish_data['condition'].unique()[0]
            genotype  = fish_data['genotype'].unique()[0]
            suffix    = f"({well}, {genotype}, {condition} mM)"
        else:
            genotype  = fish_data['genotype'].unique()[0]
            suffix    = f"({well}, {genotype})"

        # Define the palette
        palette = sns.color_palette("hls", n_colors=df['genotype'].nunique())
        # print(palette)

        # Global font size settings
        # plt.rc('axes', titlesize=18)
        # plt.rc('axes', labelsize=16)
        # plt.rc('xtick', labelsize=14)
        # plt.rc('ytick', labelsize=14)

        plt.rcParams.update({
            'axes.titlesize': 28,     # subplot titles
            'axes.labelsize': 28,     # x/y axis labels
            'xtick.labelsize': 28,    # x-axis tick labels
            'ytick.labelsize': 28,    # y-axis tick labels
            'legend.fontsize': 28,    # legend text
            'figure.titlesize': 36    # overall figure title (if used)
        })


        # Create figure with subplots
        fig, axes = plt.subplots(len(event_windows) + 1, 1, figsize=figsize)

        # This is just for the first plot
        unique_value = fish_data[hue].dropna().unique()[0]
        palette = {unique_value: '#bd2904'}

        # Plot full data in the first subplot
        sns.lineplot(ax=axes[0], x='elapsed_time', y='data1', data=fish_data, hue=hue, palette=palette)
        for start, end, _ in event_windows:
            axes[0].axvspan(start, end, color='lightblue', alpha=0.5)

        if drug_app: axes[0].axvspan(drug_app[0], drug_app[1], color='lightgray', alpha=0.3)

        axes[0].set_title(f"{title_prefix} Raw Data {suffix}")
        axes[0].set_xlabel("Elapsed Time (minutes)")
        axes[0].set_ylabel("Δ Pixel")
        axes[0].legend(title=hue, loc='best')

        if 'condition' in fish_data.columns:
            legend = axes[0].legend(title="[PTZ]")
            legend.set_title("[PTZ]", prop={'size': 28})
        else:
            legend = axes[0].legend(title="Full Trace")
            legend.set_title("Full Trace", prop={'size': 28})

        # Plot individual event windows
        for i, (start, end, title) in enumerate(event_windows):
            event_data = fish_data[(fish_data['elapsed_time'] > start) & (fish_data['elapsed_time'] < end)]
            sns.lineplot(ax=axes[i + 1], x='elapsed_time', y='data1', data=event_data)
            axes[i + 1].set_title(f"{title_prefix} {title}")# {suffix}")
            axes[i + 1].set_xlabel("Elapsed Time (minutes)")
            axes[i + 1].set_ylabel("Δ Pixel")
            # axes[i + 1].legend(title=hue, loc='best')

        plt.tight_layout(h_pad=4)

        save_path = f"{self.path}{self.name}_single_plot_{well}.svg"
        # Save as SVG
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f"Plot saved to: {save_path}")


        plt.show()

    # def detect_seizures_v1(self, fish, upper_threshold=35, lower_threshold=2, min_frame_duration=25, min_ibi=3, plot=False):
    #     # Define thresholds
    #     # upper_threshold    = 35  # Activity level for potential seizure
    #     # lower_threshold    = 2  # Defines "zero" or near-zero baseline
    #     # min_frame_duration = 25  # Minimum frames required to qualify as a seizure event
    #     # min_ibi            = 3  # Minimum interbout interval to confirm a separate event
    #
    #     # Ensure data is sorted by time (but maintain original index)
    #     fish = fish.sort_values(by="fullts", ignore_index=True)
    #
    #     # Create a boolean column indicating if data1 is above the threshold
    #     fish["above_threshold"] = fish["data1"] >= upper_threshold
    #     fish["below_threshold"] = fish["data1"] <= lower_threshold
    #
    #     # Initialize variables
    #     seizure_events = []  # Stores (start_index, end_index, duration, IBI)
    #     temp_events    = []
    #     in_seizure     = False  # Flag to track if we're currently in a seizure
    #     start_index    = None  # To store the start of the bout
    #     end_index      = None  # To store the end of the bout
    #
    #     next_start_index = None  # Tracks the next event's start
    #
    #     potential_event_end = None
    #     frame_count = 0
    #
    #     # Convert DataFrame index to a list to ensure correct referencing
    #     frame_indices = fish.index.tolist()
    #
    #     # Detect seizure events
    #
    #     for i, row in fish.iterrows():
    #         # print(i)
    #         if row["above_threshold"]:
    #             # print(f"^ this one is above threshold ({upper_threshold})")
    #             if not in_seizure:
    #                 for j in range(i, -1, -1): # Look backward for bout start
    #                     if fish.loc[j, 'below_threshold']:
    #                         start_index = j
    #                         # print("START: index=",start_index, "elapsed_time=",fishC2.loc[start_index, 'elapsed_time'])
    #                         break
    #
    #                 # Fix: If no below_threshold was found, default to i
    #                 if start_index is None:
    #                     start_index = i  # Assign i to prevent NoneType errors
    #
    #                 in_seizure  = True
    #                 frame_count = 0
    #
    #         else: # if row is not above upper threshold
    #             if in_seizure: # but still in seizure and data1 is below the lower threshold
    #
    #
    #                 if row["below_threshold"]:
    #                     # print(f"This is below the lower bound ({lower_threshold})")
    #                     if potential_event_end == None:
    #                         potential_event_end = i
    #                     else:
    #                         frame_count += 1
    #                     # print(f"This is a potential end; {frame_count}")
    #                 else:
    #                     # print('Above lower threshold count:',frame_count)
    #                     if frame_count <= min_ibi: # This means that the event is still ongoing
    #                         # duration  = potential_event_end - start_index
    #                         potential_event_end = frame_count
    #                         temp_event = (start_index, potential_event_end, frame_count, i)
    #                         temp_events.append(temp_event)
    #                         # print(f"temp_event: {temp_event}")
    #                         frame_count = 0
    #
    #                     elif len(temp_events)>0:
    #                         # print(f"temp_events: {temp_events}")
    #                         last_temp_event = temp_events.pop()
    #
    #                         end_index = last_temp_event[3]+1
    #                         duration  = end_index - start_index
    #                         ibi       = (i-1) - end_index
    #                         # print(i)
    #                         event     = (start_index, end_index, duration, ibi)
    #                         if duration > min_frame_duration:
    #                             seizure_events.append(event)
    #                         temp_events     = []
    #
    #
    #                     else:
    #
    #
    #                         end_index = potential_event_end
    #                         ibi       = frame_count
    #                         duration  = end_index - start_index
    #                         # print(temp_events)
    #
    #                         if duration > min_frame_duration:
    #                             seizure_events.append((start_index, end_index, duration, ibi))
    #                             # print(f'START: {start_index}, END: {end_index}, ibi: {ibi}')
    #                             start_index = end_index + ibi
    #
    #                         in_seizure = False
    #                         end_index = None
    #                         potential_event_end = None  # Reset temporary marker
    #                         frame_count = 0
    #                         # temp_events = []
    #
    #     #TODO
    #     # if in_seizure:
    #     #     duration = len(fishC2) - start_index
    #     #     if duration > min_frame_duration:
    #     #         ibi = start_index - last_end_index if last_end_index is not None else None
    #     #         seizure_events.append((start_index, len(fishC2) - 1, duration, ibi))
    #
    #
    #         # if i == 300: break
    #
    #     # Convert seizure events to DataFrame
    #     seizure_df = pd.DataFrame(seizure_events, columns=["start", "end", "duration", "IBI"])
    #
    #     # Convert 'duration' and 'IBI' to numeric immediately
    #     seizure_df['duration'] = pd.to_numeric(seizure_df['duration'], errors='coerce')
    #     seizure_df['IBI'] = pd.to_numeric(seizure_df['IBI'], errors='coerce')
    #
    #     if plot:
    #         self.plot_events(fish, seizure_df, frame_indices, upper_threshold, lower_threshold)
    #
    #
    #     return seizure_df

    def detect_seizures(self, fish, upper_threshold=35, lower_threshold=2, min_frame_duration=25, min_ibi=3  , plot=False):
        # Define thresholds
        # upper_threshold    = 35  # Activity level for potential seizure
        # lower_threshold    = 2  # Defines "zero" or near-zero baseline
        # min_frame_duration = 25  # Minimum frames required to qualify as a seizure event
        # min_ibi            = 3  # Minimum interbout interval to confirm a separate event

        # Ensure data is sorted by time (but maintain original index)
        fish = fish.sort_values(by="fullts", ignore_index=True)

        # Create a boolean column indicating if data1 is above the threshold
        fish["above_threshold"] = fish["data1"] >= upper_threshold
        fish["below_threshold"] = fish["data1"] <= lower_threshold

        # Initialize variables
        seizure_events = []  # Stores (start_index, end_index, duration, IBI)
        temp_events    = []
        in_seizure     = False  # Flag to track if we're currently in a seizure
        start_index    = None  # To store the start of the bout
        end_index      = None  # To store the end of the bout

        next_start_index = None  # Tracks the next event's start

        potential_event_end = None
        frame_count = 0

        # Convert DataFrame index to a list to ensure correct referencing
        frame_indices = fish.index.tolist()

        # Detect seizure events

        for i, row in fish.iterrows():
            # print(i)
            if row["above_threshold"]:
                # print(f"^ this one is above threshold ({upper_threshold})")
                if not in_seizure:
                    for j in range(i, -1, -1): # Look backward for bout start
                        if fish.loc[j, 'below_threshold']:
                            start_index = j
                            # print("START: index=",start_index, "elapsed_time=",fishC2.loc[start_index, 'elapsed_time'])
                            break

                    # Fix: If no below_threshold was found, default to i
                    if start_index is None:
                        start_index = i  # Assign i to prevent NoneType errors

                    in_seizure  = True
                    frame_count = 0

            else: # if row is not above upper threshold
                if in_seizure: # but still in seizure and data1 is below the lower threshold


                    if row["below_threshold"]:
                        # print(f"This is below the lower bound ({lower_threshold})")
                        if potential_event_end == None:
                            potential_event_end = i
                        else:
                            frame_count += 1
                        # print(f"This is a potential end; {frame_count}")
                    else:
                        # print('Above lower threshold count:',frame_count)
                        if frame_count <= min_ibi: # This means that the event is still ongoing
                            # duration  = potential_event_end - start_index
                            potential_event_end = frame_count
                            temp_event = (start_index, potential_event_end, frame_count, i)
                            temp_events.append(temp_event)
                            # print(f"temp_event: {temp_event}")
                            frame_count = 0

                        elif len(temp_events)>0:
                            # print(f"temp_events: {temp_events}")
                            last_temp_event = temp_events.pop()

                            end_index = last_temp_event[3]+1
                            duration  = end_index - start_index
                            ibi       = (i-1) - end_index
                            # print(i)
                            event     = (start_index, end_index, duration, ibi)
                            if duration > min_frame_duration:
                                seizure_events.append(event)
                            temp_events     = []


                        else:


                            end_index = potential_event_end
                            ibi       = frame_count
                            duration  = end_index - start_index
                            # print(temp_events)

                            if duration > min_frame_duration:
                                seizure_events.append((start_index, end_index, duration, ibi))
                                # print(f'START: {start_index}, END: {end_index}, ibi: {ibi}')
                                start_index = end_index + ibi

                            in_seizure = False
                            end_index = None
                            potential_event_end = None  # Reset temporary marker
                            frame_count = 0
                            # temp_events = []

        #TODO
        # if in_seizure:
        #     duration = len(fishC2) - start_index
        #     if duration > min_frame_duration:
        #         ibi = start_index - last_end_index if last_end_index is not None else None
        #         seizure_events.append((start_index, len(fishC2) - 1, duration, ibi))


            # if i == 300: break

        # Convert seizure events to DataFrame
        seizure_df = pd.DataFrame(seizure_events, columns=["start", "end", "duration", "IBI"])

        # Convert 'duration' and 'IBI' to numeric immediately
        seizure_df['duration'] = pd.to_numeric(seizure_df['duration'], errors='coerce')
        seizure_df['IBI'] = pd.to_numeric(seizure_df['IBI'], errors='coerce')

        if plot:
            self.plot_events(fish, seizure_df, frame_indices, upper_threshold, lower_threshold)


        return seizure_df

    def process_segments(self, segments, event_detection_params=None, save=True):
        # Store results in a dictionary
        segment_dfs = {}
        base_name = f"{self.name}"
        df_raw    = self.df

        for label, start, end in segments:
            seg_key = f"seg_{label}"
            full_label = f"{base_name}_{label}"
            seizure_key = f"all_seizure_df_{full_label}"
            # full_label = f"{base_name}_{label}"

            segment = df_raw[(df_raw['elapsed_time'] >= start) & (df_raw['elapsed_time'] <= end)]
            seizure_df = self.generate_all_seizure_df(event_detection_params, data=segment, label=full_label, save=save)

            segment_dfs[seg_key] = segment
            segment_dfs[seizure_key] = seizure_df

            print(f"Processed: {full_label}")

        return segment_dfs


    def generate_all_seizure_df(self, event_detection_params, data=None, label="all_seizure_df", save=False):
        print('Generating all_seizure_df...')
        if data is None:
            data = self.df

        print(event_detection_params)

        if event_detection_params is None:
            print("Default params used.")
            upper_threshold=35
            lower_threshold=2
            min_frame_duration=25
            min_ibi=3
        else:
            upper_threshold, lower_threshold, min_frame_duration, min_ibi = (event_detection_params[k] for k in ("upper_threshold", "lower_threshold", "min_frame_duration", "min_ibi"))



        seizure_data = []

        for (box, well) in data[['box', 'well']].drop_duplicates().itertuples(index=False):

            fish_data = data[(data['box'] == box) & (data['well'] == well)].copy()
            seizure_df = self.detect_seizures(fish_data, upper_threshold, lower_threshold, min_frame_duration, min_ibi)

            seizure_df['box'] = box
            seizure_df['well'] = well
            seizure_df['genotype'] = fish_data['genotype'].iloc[0]  # Assume genotype is constant for a fish

            if 'condition' in fish_data.columns:
                seizure_df['condition'] = fish_data['condition'].iloc[0]

            # Store in list
            seizure_data.append(seizure_df)

        # Combine all seizure event data into a single DataFrame
        # self.all_seizure_df = pd.concat(seizure_data, ignore_index=True)
        # print(Done, all_seizure_df is )
        # return self.all_seizure_df

        if data is None:
            # Combine all seizure event data into a single DataFrame
            self.all_seizure_df = pd.concat(seizure_data, ignore_index=True)
            print("Done, all_seizure_df is now an attribute of the object.")
            return self.all_seizure_df
        else:
            all_seizure_df = pd.concat(seizure_data, ignore_index=True)
            if save:
                all_seizure_df.to_csv(f'{self.path}{label}.csv', index=False)
            print("Done")
            return all_seizure_df

    def plot_events(self, fish, seizure_df, frame_indices, upper_threshold, lower_threshold):
        # Select 5 random seizure events (if available)
        num_events_to_plot = min(4, len(seizure_df))
        selected_events = seizure_df.sample(n=num_events_to_plot, random_state=42)
        # selected_events = seizure_df.loc[:10]
        print(f"Number of Seizure Events detected: {len(seizure_df)} ({fish['well'][0]}, {fish['genotype'][0]})")

        # Plot selected seizure events
        for _, event in selected_events.iterrows():
            start, end = event["start"], event["end"]

            # Extend the window by 5 frames before and after the event
            start_idx = max(frame_indices.index(start) - 20, 0)
            end_idx = min(frame_indices.index(end) + 20, len(frame_indices) - 1)

            plot_start = start_idx
            plot_end = end_idx

            # Extract subset of data for plotting
            subset = fish.loc[plot_start:plot_end]

            # Plot the selected event window
            plt.figure(figsize=(16, 8))

            plt.rcParams.update({
                'axes.titlesize': 30,     # subplot titles
                'axes.labelsize': 30,     # x/y axis labels
                'xtick.labelsize': 30,    # x-axis tick labels
                'ytick.labelsize': 30,    # y-axis tick labels
                'legend.fontsize': 28,    # legend text
                'figure.titlesize': 36    # overall figure title (if used)
            })

            plt.plot(subset.index, subset["data1"], color="blue", label="Activity Data")
            plt.axvspan(start, end, color="red", alpha=0.1, label="'Potential Seizure' Event")

            # Formatting
            plt.axhline(y=upper_threshold, color='black', linestyle='--', label=f"Threshold ({upper_threshold})")
            plt.axhline(y=lower_threshold, color='gray', linestyle=':', label=f"Baseline ({lower_threshold})")
            plt.xlabel("Frame Index")
            plt.ylabel("Δ Pixel")
            plt.title(f"'Potential Seizure' Event from Frame {start} to {end} ({event['duration']}, {fish['well'][0]}, {fish['genotype'][0]})", pad=25)
            plt.legend()


            save_path = f"{self.path}{self.name}_event_{start}_{end}_{fish['well'][0]}.svg"
            # Save as SVG
            plt.savefig(save_path, format='svg', bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

            plt.show()

    # def detect_normal_swim_bouts(self, fish, upper_threshold=35, lower_threshold=2, min_frame_duration=25, min_ibi=3, plot=False):
    #
    #
    #
    #     return normal_bouts_df

class MiddurData(Experiment): #The output is not compatible with sleep analysis
    # def __init__(self, date, box1, box2, exp, export=False):
    def __init__(self, date, box1, box2, exp, export=False, nbox=2, cbox=None, omit=None):
        print('Initialising MiddurData...')
        # Call the parent class's __init__
        super().__init__(date, box1, box2, exp, export, omit=omit)


        if self.omit is None:
            self.prepped_data = self.prepare_raw_data()
        else:
            self.prepped_data, self.prepped_filtered_data = self.prepare_raw_data()




    def prepare_raw_data(self):

        genotype_df = self.get_genotype_df()

        # Columns to read from the CSV file
        cols = ["location","start", "end", "animal", "stdate", "sttime", "middur"]

        # Read and add box identifiers
        raw  = pd.read_csv(self.filename, usecols=cols, low_memory=False, header=0, parse_dates=[['stdate', 'sttime']])

        # Adjust Box column based on the 'animal' prefix
        raw['box'] = raw['animal'].str.startswith('1-').replace({True: self.box1, False: self.box2})

        ###

        if self.cond_map_files:
            condition_df = self.get_condition_df()


        # Split 'animal' into 'plate' and 'well'
        raw[['plate', 'well']] = raw['animal'].str.extract(r'(\d)-([A-H]\d{2})')
        raw['plate'] = raw['plate'].astype(int)
        # Remove leading zeros from 'well' column numbers
        raw['well'] = raw['well'].str.replace(r'([A-H])0(\d)', r'\1\2', regex=True)
        #
        if self.cond_map_files:
            # Merge data on Location and Box
            merged_data = raw.merge(genotype_df, on=['box', 'well'], how='left').merge(condition_df, on=['box', 'well'], how='left')
        else:
            merged_data = raw.merge(genotype_df, on=['box', 'well'], how='left')

        # # Set WT as default genotype where missing
        # merged_data['genotype'].fillna('WT', inplace=True)
        merged_data['genotype'].dropna(inplace=True)


        merged_data = merged_data[~merged_data['genotype'].isin(['empty', 'NaN'])]

        prepped_filtered_data = self.filter_df_by_omit(merged_data, self.omit)

        # # Organize columns
        prepped_complete_data = merged_data
        #
        # # Calculate the 'CLOCK' column based on the combined 'stdate' and 'sttime'
        # prepped_data['clock'] = self.calculate_clock(prepped_data['stdate_sttime'])
        print('Done.')

        # self.correct_start_time = prepped_complete_data['stdate_sttime'].min()

        if self.omit is None:
            return prepped_complete_data
        else:
            return prepped_complete_data, prepped_filtered_data

    # Filtering logic
    def filter_df_by_omit(self, df, omit):
        if omit is None:
            # If omit is None, return the original DataFrame
            print("No filtering applied: 'omit' is None.")
            return df

        # Create a mask to exclude rows matching the omit criteria
        mask = df.apply(lambda row: row['well'] in omit.get(row['box'], []), axis=1)
        return df[~mask]  # Keep rows where mask is False


    # def sanitise_data(self, dirty_data): # Extract pertinent data from CSV and define dtype for each. Returns array and df for fun.
    #     # Get unique animals (larvae) in the dataset
    #     return None

    def quick_plot_per_fish(self):
        print("Drawing quick plots...")
        data = self.prepped_data

        # Set the reference time (e.g., the earliest timestamp)
        # reference_time = data['stdate_sttime'].min()
        #
        # # Calculate elapsed time in seconds (or use .total_seconds() for float)
        # data['elapsed_time'] = (data['stdate_sttime'] - reference_time).dt.total_seconds()/60

        # data = data[data['elapsed_time'] >= 12]
        #
        # print(data)

        # Get unique boxes from the dataset
        unique_boxes = sorted(data['box'].unique())

        # Plot data for each box
        for box_id in unique_boxes:
            self.plot_box_data(data, box_id)

        print("Quick plots, done.")
        return None

    # Function to plot data for a single box
    def plot_box_data(self, data, box_id):
        # data = self.prepped_data
        box_data = data[data['box'] == box_id].copy()
        print(box_data)

        if self.omit is not None and box_id in self.omit:
            omit_list = self.omit[box_id]  # Get the list for the corresponding box_id
            print(f"Wells to omit for box {box_id}: {omit_list}")
        else:
            omit_list = []
            print(f"No wells to omit for box {box_id}.")

        # Check if the condition column exists
        condition_exists = 'condition' in box_data.columns

        # Ensure well positions are uppercase for consistency
        box_data['well'] = box_data['well'].str.upper()
        box_data['genotype'] = box_data['genotype'].str.lower()

        # Count the number of unique animals for each genotype in this box
        genotype_counts = box_data.groupby('genotype')['animal'].nunique()

        # Define the 96-well plate layout
        well_order = [f"{row}{col}" for row in 'ABCDEFGH' for col in range(1, 13)]


        # If condition exists, set up the condition palette
        if condition_exists:
            unique_conditions = box_data['condition'].dropna().unique()
            condition_palette = sns.hls_palette(n_colors=len(unique_conditions))
            condition_map = dict(zip(unique_conditions, condition_palette))



        # Define the custom color palette and genotype order
        new_colors_order = sns.color_palette(palette='Set2', n_colors=3)
        new_colors_order[0], new_colors_order[1], new_colors_order[2] = (
            new_colors_order[1],
            new_colors_order[0],
            new_colors_order[2],
        )
        color_map = dict(zip(['wt', 'het', 'hom'], new_colors_order))

        # Initialize the figure and axes for the grid
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(8, 12, figsize=(20, 15), sharex=True, sharey=True)
        fig.suptitle(
            f"Quick Glance Activity Traces for {self.exp} Zebrafish in Box {box_id} ({self.name})",
            fontsize=16,
        )

        # Iterate through each subplot (well position) and plot data if available
        for i, well in enumerate(well_order):

            row, col = divmod(i, 12)
            ax = axes[row, col]

            well_data = box_data[box_data['well'] == well]
            if not well_data.empty:
                genotype = well_data['genotype'].iloc[0] if 'genotype' in well_data else 'wt'
                if well in omit_list:
                    color = 'lightgray'
                else:
                    color = color_map.get(genotype, 'gray')  # Default to gray if genotype is missing

                # Shade the background if condition exists
                if condition_exists:
                    condition = well_data['condition'].iloc[0] if 'condition' in well_data else 'unknown'
                    condition_color = condition_map.get(condition, 'lightgray')  # Default to light gray if condition is missing
                    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=condition_color, alpha=0.1))
                    label = f"{well}\n{condition}"
                else:
                    label = well


                ax.plot(well_data['start'], well_data['middur'], color=color, label=f"{genotype}")
                # ax.set_title(well, fontsize=8)
                ax.set_title(label, fontsize=8)  # Label with well and condition

            # Turn off axis for cleaner presentation
            # ax.axis('off')

        # Add a legend for genotype colors with counts
        handles = [
            plt.Line2D(
                [0], [0], color=color, lw=2,
                label=f"{genotype.upper()} ({genotype_counts.get(genotype, 0)})"
            )
            for genotype, color in color_map.items()
        ]
        fig.legend(handles=handles, loc='upper right', title="Genotype (Count)")

        # Adjust layout for better visualization
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


        return None

    def plot_genotype_activity(self, bin_minutes=None, colors=None, use_filtered=False):
            """
            Plot mean middur ± SEM for each genotype over time.

            Args:
                bin_minutes: Optional int to bin time into larger intervals for smoother traces.
                colors: Optional dict mapping genotype to color.
                use_filtered: If True and self.omit is set, use prepped_filtered_data.
            """

            if use_filtered and self.omit is not None:
                data = self.prepped_filtered_data.copy()
            else:
                data = self.prepped_data.copy()

            # Drop NaN genotypes and empties
            data = data[data['genotype'].notna() & ~data['genotype'].isin(['empty', 'NaN'])]

            # if colors is None:
            #     new_colors_order = sns.color_palette(palette='Set2', n_colors=3)
            #     new_colors_order[0], new_colors_order[1], new_colors_order[2] = (
            #         new_colors_order[1], new_colors_order[0], new_colors_order[2],
            #     )
            #     colors = dict(zip(['wt', 'het', 'hom'], new_colors_order))
            if colors is None:
                colors = {"wt": "#FFA500", "het": "#228B22", "hom": "#1E90FF"}

            genotypes = [g for g in ['wt', 'het', 'hom'] if g in data['genotype'].str.lower().unique()]

            fig, ax = plt.subplots(figsize=(10, 5))

            for geno in genotypes:
                subset = data[data['genotype'].str.lower() == geno].copy()
                if subset.empty:
                    continue

                n_animals = subset['animal'].nunique()

                if bin_minutes:
                    bin_size = bin_minutes * 60
                    subset['time_bin'] = (subset['start'] // bin_size) * bin_size
                    # Sum middur within each bin per animal, then average across animals
                    grouped = subset.groupby(['time_bin', 'animal'])['middur'].sum().reset_index()
                    stats = grouped.groupby('time_bin')['middur'].agg(['mean', 'sem']).reset_index()
                    x = stats['time_bin'] / 3600
                else:
                    stats = subset.groupby('start')['middur'].agg(['mean', 'sem']).reset_index()
                    x = stats['start'] / 3600

                color = colors.get(geno, 'gray')
                ax.plot(x, stats['mean'], label=f'{geno.upper()} (n={n_animals})',
                        color=color, linewidth=1.2)
                ax.fill_between(x, stats['mean'] - stats['sem'], stats['mean'] + stats['sem'],
                                alpha=0.1, color=color)

            max_hours = data['start'].max() / 3600
            ax.axvspan(14, max_hours, alpha=0.08, color='gray', label='Dark Period')
            # ax.axvspan(1, 1 + 5/60, alpha=0.10, color='yellow', label='Light Stimulus Cycle')
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Average Activity (middur; s/min)')
            ax.legend()
            total_mins = int(data['start'].max() / 60)
            ax.set_title(f'Average Activity by Genotype — {self.exp} ({self.name}) [{total_mins} min]')
            sns.set_theme(style="whitegrid")
            plt.tight_layout()
            plt.show()

    def analyse_auc(self, use_filtered=False, epoch_bounds=None, colors=None):
        """
        Compare genotypes using Area Under Curve.

        Args:
            use_filtered: If True and self.omit is set, use prepped_filtered_data.
            epoch_bounds: Optional dict of named epochs, e.g.
                          {'baseline': (0, 3600), 'stimulus': (3600, 3900), 'post': (3900, 7200)}
                          Values in seconds. If None, uses full experiment as one epoch.
            colors: Optional dict mapping genotype to color.
        """
        from scipy import stats as sp_stats

        if use_filtered and self.omit is not None:
            data = self.prepped_filtered_data.copy()
        else:
            data = self.prepped_data.copy()

        data = data[data['genotype'].notna() & ~data['genotype'].isin(['empty', 'NaN'])]
        data['genotype'] = data['genotype'].str.lower()

        if colors is None:
            colors = {"wt": "#FFA500", "het": "#228B22", "hom": "#1E90FF"}

        genotypes = [g for g in ['wt', 'het', 'hom'] if g in data['genotype'].unique()]

        if epoch_bounds is None:
            epoch_bounds = {'full': (data['start'].min(), data['start'].max())}

        n_epochs = len(epoch_bounds)
        fig, axes = plt.subplots(1, n_epochs, figsize=(5 * n_epochs, 5), squeeze=False)
        axes = axes[0]

        results = {}

        for idx, (epoch_name, (t_start, t_end)) in enumerate(epoch_bounds.items()):
            ax = axes[idx]
            epoch_data = data[(data['start'] >= t_start) & (data['start'] <= t_end)]

            # Calculate AUC per animal using trapezoidal rule
            auc_per_animal = (
                epoch_data.groupby(['genotype', 'animal'])
                .apply(lambda g: np.trapz(g['middur'], g['start']))
                .reset_index(name='auc')
            )

            # Statistical test
            groups = [grp['auc'].values for _, grp in auc_per_animal.groupby('genotype') if len(grp) > 1]
            group_labels = [name for name, grp in auc_per_animal.groupby('genotype') if len(grp) > 1]

            # Normality check
            all_normal = all(
                sp_stats.shapiro(g)[1] > 0.05 for g in groups if len(g) >= 3
            )

            if all_normal and len(groups) >= 2:
                stat, p = sp_stats.f_oneway(*groups)
                test_name = 'One-way ANOVA'
            else:
                stat, p = sp_stats.kruskal(*groups)
                test_name = 'Kruskal-Wallis'

            results[epoch_name] = {
                'test': test_name, 'statistic': stat, 'p_value': p,
                'auc_data': auc_per_animal
            }

            # Post-hoc pairwise if significant
            pairwise = {}
            if p < 0.05 and len(groups) >= 2:
                from itertools import combinations
                pairs = list(combinations(group_labels, 2))
                for g1, g2 in pairs:
                    a = auc_per_animal[auc_per_animal['genotype'] == g1]['auc']
                    b = auc_per_animal[auc_per_animal['genotype'] == g2]['auc']
                    _, pw_p = sp_stats.mannwhitneyu(a, b, alternative='two-sided')
                    pairwise[f'{g1} vs {g2}'] = pw_p
                # Bonferroni correction
                n_comp = len(pairwise)
                pairwise = {k: min(v * n_comp, 1.0) for k, v in pairwise.items()}
                results[epoch_name]['pairwise'] = pairwise

            # Post-hoc pairwise (always run)
            # from itertools import combinations
            # pairwise = {}
            # pairs = list(combinations(group_labels, 2))
            # for g1, g2 in pairs:
            #     a = auc_per_animal[auc_per_animal['genotype'] == g1]['auc']
            #     b = auc_per_animal[auc_per_animal['genotype'] == g2]['auc']
            #     _, pw_p = sp_stats.mannwhitneyu(a, b, alternative='two-sided')
            #     pairwise[f'{g1} vs {g2}'] = pw_p
            # # Bonferroni correction
            # n_comp = len(pairwise)
            # pairwise = {k: min(v * n_comp, 1.0) for k, v in pairwise.items()}
            # results[epoch_name]['pairwise'] = pairwise

            # Plot
            for geno in genotypes:
                geno_auc = auc_per_animal[auc_per_animal['genotype'] == geno]['auc']
                color = colors.get(geno, 'gray')
                positions = [genotypes.index(geno)]
                bp = ax.boxplot(geno_auc, positions=positions, widths=0.5,
                               patch_artist=True, showfliers=False)
                bp['boxes'][0].set_facecolor(color)
                bp['boxes'][0].set_alpha(0.4)
                # Overlay individual points
                jitter = np.random.normal(0, 0.05, size=len(geno_auc))
                ax.scatter(np.array(positions * len(geno_auc)) + jitter, geno_auc,
                          color=color, alpha=0.7, s=20, zorder=3)

            ax.set_xticks(range(len(genotypes)))
            ax.set_xticklabels([g.upper() for g in genotypes])
            ax.set_ylabel('AUC (middur × seconds)')
            ax.set_title(f'{epoch_name}\n{test_name}: p={p:.4f}')

            # Add significance bars for pairwise
            if 'pairwise' in results[epoch_name]:
                y_max = ax.get_ylim()[1]
                step = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08
                for j, (pair, pw_p) in enumerate(results[epoch_name]['pairwise'].items()):
                    g1, g2 = pair.split(' vs ')
                    x1, x2 = genotypes.index(g1), genotypes.index(g2)
                    y = y_max + step * (j + 1)
                    star = '***' if pw_p < 0.001 else '**' if pw_p < 0.01 else '*' if pw_p < 0.05 else 'ns'
                    ax.plot([x1, x2], [y, y], 'k-', linewidth=1)
                    ax.text((x1 + x2) / 2, y, star, ha='center', va='bottom', fontsize=10)
                    # ax.set_ylim(ax.get_ylim()[0], y_max + step * (len(results[epoch_name].get('pairwise', {})) + 2))
        fig.suptitle(f'AUC Analysis — {self.exp} ({self.name})', fontsize=14)
        plt.tight_layout()
        plt.show()

        # Print results summary
        for epoch_name, res in results.items():
            print(f"\n--- {epoch_name} ---")
            print(f"  {res['test']}: stat={res['statistic']:.4f}, p={res['p_value']:.4f}")
            if 'pairwise' in res:
                for pair, pw_p in res['pairwise'].items():
                    print(f"  {pair}: p={pw_p:.4f} (Bonferroni corrected)")

        return results


    def analyse_lmm(self, use_filtered=False, bin_minutes=None, epoch_bounds=None):
        """
        Linear mixed-effects model: middur ~ genotype * epoch + (1 | animal).

        Args:
            use_filtered: If True and self.omit is set, use prepped_filtered_data.
            bin_minutes: Optional int to bin time before modelling (reduces data size).
            epoch_bounds: Optional dict of named epochs in seconds, e.g.
                          {'baseline': (0, 3600), 'stimulus': (3600, 3900), 'post': (3900, 7200)}.
                          If None, genotype effect is tested across all time with no epoch interaction.
        """
        import statsmodels.formula.api as smf

        if use_filtered and self.omit is not None:
            data = self.prepped_filtered_data.copy()
        else:
            data = self.prepped_data.copy()

        data = data[data['genotype'].notna() & ~data['genotype'].isin(['empty', 'NaN'])]
        data['genotype'] = data['genotype'].str.lower()

        # Optional time binning
        if bin_minutes:
            bin_size = bin_minutes * 60
            data['time_bin'] = (data['start'] // bin_size) * bin_size
            data = data.groupby(['animal', 'genotype', 'time_bin']).agg(
                middur=('middur', 'sum')
            ).reset_index()
            data.rename(columns={'time_bin': 'start'}, inplace=True)

        # Assign epochs if provided
        if epoch_bounds:
            def assign_epoch(t):
                for name, (t_start, t_end) in epoch_bounds.items():
                    if t_start <= t <= t_end:
                        return name
                return None
            data['epoch'] = data['start'].apply(assign_epoch)
            data = data[data['epoch'].notna()]

            # Set reference categories
            data['genotype'] = pd.Categorical(data['genotype'], categories=['wt', 'het', 'hom'])
            epoch_names = list(epoch_bounds.keys())
            data['epoch'] = pd.Categorical(data['epoch'], categories=epoch_names)

            # Fit model with interaction
            formula = 'middur ~ C(genotype) * C(epoch)'
            print(f"Fitting LMM: {formula} + (1 | animal)")
            model = smf.mixedlm(formula, data, groups=data['animal'])
            result = model.fit()
            print(result.summary())

            # Also fit per-epoch for clearer interpretation
            print("\n--- Per-epoch models ---")
            for epoch_name in epoch_names:
                epoch_data = data[data['epoch'] == epoch_name]
                formula_epoch = 'middur ~ C(genotype)'
                print(f"\n  Epoch: {epoch_name}")
                print(f"  Fitting: {formula_epoch} + (1 | animal)")
                model_epoch = smf.mixedlm(formula_epoch, epoch_data, groups=epoch_data['animal'])
                result_epoch = model_epoch.fit()
                print(result_epoch.summary())

        else:
            # Simple model: genotype effect across all time
            data['genotype'] = pd.Categorical(data['genotype'], categories=['wt', 'het', 'hom'])

            formula = 'middur ~ C(genotype)'
            print(f"Fitting LMM: {formula} + (1 | animal)")
            model = smf.mixedlm(formula, data, groups=data['animal'])
            result = model.fit()
            print(result.summary())

        return result

class MiddurData_SA(Experiment): #The output is compatible with sleep analysis
    # def __init__(self, date, box1, box2, exp, export=False):
    def __init__(self, date, box1, box2, exp, export=False):

        # Call the parent class's __init__
        super().__init__(date, box1, box2, exp, export)

        self.for_sleep_analysis = True

    def prepare_raw_data(self): # Iterates through files, cleans using sanitise_data, appends each well as a matrix (this_well) into 3D ndarray (all_wells). Returns keyed plates dict with each plate ndarray stored inside.
        # Columns to read from the CSV file
        cols = ["start", "end", "animal", "stdate", "sttime", "middur"]

        # Read the CSV file with specific columns, header starts at row 1, and combine 'stdate' and 'sttime'
        dirty_data = pd.read_csv(self.path + self.name + ".csv", usecols=cols, low_memory=False, header=0, parse_dates=[['stdate', 'sttime']])

        # Convert big old xls in terminal:
        # ssconvert large_file.xls large_file.csv
        # dirty_data = pd.read_excel(self.path + self.name + ".xls", usecols=cols, header=0, parse_dates=[['stdate', 'sttime']], engine='xlrd')

        cleaned_df = self.sanitise_data(dirty_data)

        if self.export:
            self.export_to_txt(cleaned_df, self.date, self.path)

        print('Finished.')
        return cleaned_df

    def sanitise_data(self, dirty_data): # Extract pertinent data from CSV and define dtype for each. Returns array and df for fun.
        # Get unique animals (larvae) in the dataset
        larvae = sorted(dirty_data["animal"].unique())

        # Initialize a dictionary to store the middur data keyed by 'FISHx'
        middur_dict = {}

        # Loop over each larva to extract middur data
        for i, larva in enumerate(larvae):
            # Extract data for the current larva
            larvalData = dirty_data[dirty_data["animal"] == larva]

            # Add 'start' and 'end' only once (for the first larva)
            if i == 0:
                middur_dict['start'] = larvalData['start'].reset_index(drop=True)
                middur_dict['end'] = larvalData['end'].reset_index(drop=True)


                # Calculate the 'CLOCK' column based on the combined 'stdate' and 'sttime'
                clock = self.calculate_clock(larvalData['stdate_sttime']).reset_index(drop=True)

            # Key the middur data by the fish name 'FISHx'
            fishName = f'FISH{i+1}'
            middur_dict[fishName] = larvalData['middur'].reset_index(drop=True)


            # If we want genotype data, uncomment the next two lines and comment out above line
            # genotype = pd.Series([fish_genotype_dict[i+1]])
            # middur_dict[fishName] = pd.concat([genotype,larvalData['middur'].reset_index(drop=True)])

        # Convert the dictionary into a DataFrame
        middur_df = pd.DataFrame(middur_dict)

        #### Legacy (and a bit redundant) adjustments to make it compatible with the existing analysis ####
        middur_df = self.legacy(middur_df, larvae)

        middur_df['CLOCK'] = clock

        return middur_df


    # # Function to calculate the 'CLOCK' column
    # def calculate_clock(self, series):
    #     def compute_clock(start_datetime):
    #         reference_time = start_datetime.replace(hour=9, minute=0, second=0)
    #         if start_datetime < reference_time:
    #             reference_time -= timedelta(days=1)
    #         return (start_datetime - reference_time).total_seconds() / 3600
    #
    #     return pd.concat([pd.Series([""]), series.apply(compute_clock)])

    def legacy(self, middur_df, larvae):

        # Rename the 'start' and 'end' columns for compatibility
        middur_df.rename(columns={'start': 'TIME(SECONDS)', 'end': 'NA'}, inplace=True)

        # Create the second row after the header ('start', 'end', 'middur' repeated for each fish)
        header_row = ['start', 'end'] + ['middur'] * len(larvae)

        # Create a DataFrame for the additional row
        additional_row_df = pd.DataFrame([header_row], columns=middur_df.columns)

        # Concatenate the additional row with the original DataFrame
        final_df = pd.concat([additional_row_df, middur_df], ignore_index=True)

        # Append two blank columns, both labeled 'NA'
        final_df['NA_1'] = ""  # First blank column
        final_df['NA_2'] = ""  # Second blank column

        # Rename both columns to 'NA'
        final_df.columns = [col if col not in ['NA_1', 'NA_2'] else 'NA' for col in final_df.columns]

        return final_df

    # Function to export the DataFrame as a .txt file
    def export_to_txt(self, final_df, date, path):
        filename = os.path.join(path, f"{date}_00_DATA.txt")  # Construct the file path
        final_df.to_csv(filename, sep='\t', index=False)  # Export DataFrame to .txt with tab separators
        print(f"File saved at: {filename}")  # Confirmation message

    def temp_add_genotype(self):

        # Load the genotype file
        genotype_file_path = self.path+self.date+'_00genotype.txt'
        genotype_data = pd.read_csv(genotype_file_path, sep='\t', header=None)

        # Initialize an empty dictionary for genotypes, defaulting to 'unknown'
        fish_genotype_dict = {fish_id: 'unknown' for fish_id in range(1, 193)}

        # Process rows from the genotype file starting from the third row
        for row in genotype_data.iloc[2:].itertuples(index=False):
            # Fish IDs in each of the genotype columns (wt, hom, het)
            if pd.notna(row[0]):
                fish_genotype_dict[int(float(row[0]))] = 'wt'
            if pd.notna(row[1]):
                fish_genotype_dict[int(float(row[1]))] = 'hom'
            if pd.notna(row[2]):
                fish_genotype_dict[int(float(row[2]))] = 'het'

        geno_table = pd.DataFrame.from_dict(fish_genotype_dict, orient='index', columns=['Genotype'])
        # Display the final ordered fish genotype dictionary
        # print(fish_genotype_dict)
        return geno_table

    def temp_merge_geno_with_data(self, activity_df, genotype_df):
        # This is for merging the sleep analysis output df with a geno table so that the genotypes become headers in the third row

        # Ensure column names are unique
        activity_df.columns = pd.Index(activity_df.columns.map(str))  # Ensure column names are strings

        # Step 1: Map genotypes to the columns
        genotype_map = {f"FISH{fish_id}": genotype for fish_id, genotype in genotype_df['Genotype'].items()}

        # Step 2: Create a row of genotypes for each column
        genotype_row = [genotype_map.get(col, 'unknown') if 'FISH' in col else '' for col in activity_df.columns]

        # Step 3: Insert the genotype row as a new DataFrame
        genotype_row_df = pd.DataFrame([genotype_row], columns=activity_df.columns)

        # Step 4: Insert the genotype row as the third row
        merged_df = pd.concat([activity_df.iloc[:1], genotype_row_df, activity_df.iloc[1:]], ignore_index=True)

        return merged_df

class KASP():

    # USAGE
    # box1 = 16
    # box2 = 17
    # csv_files = {
    #     box1:'KASP/250217_0214_16_PLPBP_7dpf_01_Genotyping Result_20250217_195245.csv',
    #     box2:'KASP/250217_0214_17_PLPBP_7dpf_Genotyping Result_20250217_195341.csv'
    # }
    # # For plate 1 and plate 2 respectively.
    # omitted_wells = {box1: ['H9','H12','G3'],
    #                  box2: ['G12','F12']}
    # drop = {box1: [], box2: []}
    # display_list = False
    #
    # results = b6.KASP(csv_files, omitted_wells, drop, display_list)

    def __init__(self, files, omitted_wells=None, drop_wells=None, controls=None, display_list=False, direct=False):
        self.files     = files
        self.omitted_wells = {box: set(wells) for box, wells in (omitted_wells or {14:{}}).items()}
        self.drop_wells    = {box: set(wells) for box, wells in (drop_wells or {14:{}}).items()}
        self.controls      = {box: set(wells) for box, wells in (controls or {14:{}}).items()}
        self.display_list  = display_list
        self.direct        = direct # This is True if the genotype data is manual (not from the auto-gen CSV from Thermofischer)

        # Dictionary to store processed data for each plate
        self.plates = {}

        # Load and process each file
        for box_id, file_path in self.files.items():
            self.plates[box_id] = self.load_data(box_id)
            # print(self.plates[box_id])
            self.plot_allele_and_well_plate(box_id)
            if display_list:
                self.print_grouped_well_lists(box_id, self.plates[box_id])


    def load_data(self, box_id):
        """Loads and preprocesses the data from the CSV file."""
        if self.files[box_id].endswith('.eds'):
            data = self.read_eds(self.files[box_id])
            data = data.dropna(axis=0)
            return data[~data['Well Position'].isin(self.drop_wells[box_id])]
        else:
            if self.direct:
                data = pd.read_csv(self.files[box_id],
                                   usecols=["Well Position", "Genotype"])
                return data
            else:
                data = pd.read_csv(self.files[box_id], skiprows=23,
                               usecols=["Well", "Well Position", "Sample", "Allele 1", "Allele 2", "Call"])
                return data[~data['Well Position'].isin(self.drop_wells[box_id])]

    def perform_clustering(self, data, box_id):
        """Clusters the data into WT, HET, and HOM using K-means."""

        # Define omitted wells before clustering
        valid_wells = ~data['Well Position'].isin(self.omitted_wells[box_id])
        print("Omitted: ",self.omitted_wells[box_id])

        # Extract only valid data for clustering
        X = data.loc[valid_wells, ['Allele 1', 'Allele 2']]

        X_scaled = StandardScaler().fit_transform(X)
        kmeans   = KMeans(n_clusters=3, n_init=10, random_state=42)
        initial_clusters = kmeans.fit_predict(X_scaled)  # Store initial cluster assignments

        # Store results only for valid wells
        data.loc[valid_wells, 'Cluster'] = initial_clusters

        # Compute the centroid distances to determine labels
        centroids = kmeans.cluster_centers_

        ref_point = np.array([X_scaled[:, 0].max(), X_scaled[:, 1].min()]) # Ref set as x max, y min aka WT
        distances = cdist(centroids, [ref_point])
        sorted_clusters = np.argsort(distances[:, 0])

        # Dynamically assign genotype labels
        cluster_labels = {sorted_clusters[0]: 'WT', sorted_clusters[1]: 'HET', sorted_clusters[2]: 'HOM'}

        # Assign correct Genotypes only for valid wells
        data.loc[valid_wells, 'Genotype'] = data.loc[valid_wells, 'Cluster'].map(cluster_labels)

        # Ensure omitted wells have NaN for cluster & labels
        data.loc[~valid_wells, ['Cluster', 'Genotype']] = np.nan



    def plot_allele_and_well_plate(self, box_id):

        data = self.plates[box_id]

        """Generates allele discrimination and 96-well plate plots."""
        if not self.direct:
            self.perform_clustering(data, box_id)
        file_name = Path(self.files[box_id]).stem
        print(file_name)

        updated_colors = {'WT': '#ff7f0e', 'HET': '#2ca02c', 'HOM': '#1f77b4'}
        data['Color'] = data['Genotype'].map(updated_colors)

        # print(self.data[['Well Position','Call','Cluster','Genotype','Color']])

        # Set up the figure with two subplots (for Allele plot and 96-well visualization)
        fig, axs = plt.subplots(1, 2, figsize=(20, 6))

        if not self.direct:
            # Allele Discrimination Plot
            ax1 = axs[0]
            for index, row in data.iterrows():
                well_pos = row['Well Position']
                color = 'black' if well_pos in self.omitted_wells[box_id] else updated_colors[row['Genotype']]
                ax1.scatter(row['Allele 1'], row['Allele 2'], color=color)
            for i, txt in enumerate(data['Well Position']):
                ax1.annotate(txt, (data['Allele 1'].iloc[i], data['Allele 2'].iloc[i]), fontsize=8)
            ax1.set(title=f'{file_name}: Allele Discrimination Plot', xlabel='Allele 1', ylabel='Allele 2')
            ax1.grid(True)

        # 96-Well Plate Visualization
        ax2 = axs[1]
        rows, columns = list('ABCDEFGH'), list(range(1, 13))
        for row_label in rows:
            for col_label in columns:
                well_pos = f"{row_label}{col_label}"
                color = 'white'  # Default for empty wells
                if well_pos in data['Well Position'].values:
                    row = data[data['Well Position'] == well_pos].iloc[0]
                    if not self.direct:
                        color = 'black' if well_pos in self.omitted_wells[box_id] else updated_colors[row['Genotype']]
                ax2.add_patch(mpatches.Rectangle((col_label - 1, rows.index(row_label)), 1, 1, facecolor=color, edgecolor='black'))
        ax2.set(xlim=(0, 12), ylim=(0, 8), xticks=np.arange(12) + 0.5, yticks=np.arange(8) + 0.5,
                xticklabels=columns, yticklabels=rows, title=f'{file_name}: 96-Well Plate Calls')
        ax2.invert_yaxis()

        # Legend
        legend_patches = [mpatches.Patch(color=updated_colors[label], label=label) for label in ['HOM', 'HET', 'WT']]
        if self.omitted_wells[box_id]:
            legend_patches.append(mpatches.Patch(color='black', label='Omitted'))
        legend_patches.append(mpatches.Patch(color='white', label='Empty'))
        ax2.legend(handles=legend_patches, loc='upper right')

        plt.tight_layout()

        plt.savefig('kasp.png', transparent=True)

        plt.show()

    def get_grouped_well_lists(self, box_id, data):
        """
        Groups wells into WT, HET, HOM, Omitted, and Dropped categories using the object's attributes.
        Returns the lists as formatted dictionaries.
        """
        all_wells = set(data['Well Position'])  # All wells in the dataset
        omitted_set  = set(self.omitted_wells[box_id]) if self.omitted_wells[box_id] else set()
        dropped_set  = set(self.drop_wells[box_id]) if self.drop_wells[box_id] else set()
        controls_set = set(self.controls[box_id]) if self.controls[box_id] else set()
        print(self.controls[box_id], box_id)

        # Exclude omitted and dropped wells from WT, HET, HOM
        valid_wells = all_wells - omitted_set - dropped_set - controls_set
        group_wells = {
            'WT': data[(data['Genotype'] == 'WT') & (data['Well Position'].isin(valid_wells))]['Well Position'].tolist(),
            'HET': data[(data['Genotype'] == 'HET') & (data['Well Position'].isin(valid_wells))]['Well Position'].tolist(),
            'HOM': data[(data['Genotype'] == 'HOM') & (data['Well Position'].isin(valid_wells))]['Well Position'].tolist(),
            'Omitted': list(omitted_set & all_wells),
            'Dropped': list(dropped_set),
            'Controls': list(controls_set)
        }

        # Format and sort the output for each group
        for group, wells in group_wells.items():
            formatted_wells = {}
            for well in wells:
                row = well[0]
                col = int(well[1:])  # Convert column to integer for numerical sorting
                if row not in formatted_wells:
                    formatted_wells[row] = []
                formatted_wells[row].append(col)
            # Sort columns numerically for each row
            group_wells[group] = {row: sorted(cols) for row, cols in formatted_wells.items()}

        # # To get grouped wells as a dictionary
        # grouped_wells = results.get_grouped_well_lists()
        # print(grouped_wells)

        return group_wells

    def print_grouped_well_lists(self, box_id, data):
        """
        Prints grouped and formatted well lists for WT, HET, HOM, Omitted, and Dropped categories.
        """

        grouped_wells = self.get_grouped_well_lists(box_id, data)
        print('Genotypes for plate:',box_id)
        for group, wells in grouped_wells.items():
            print(f"{group}:")
            for row, cols in sorted(wells.items()):  # Sort rows alphabetically
                print(f"  {row}: {', '.join(map(str, cols))}")

        # Load and process each file
        # for box_id, data in self.plates.items():
        #     grouped_wells = self.get_grouped_well_lists(box_id, data)
        #     print('Genotypes for plate:',box_id)
        #     for group, wells in grouped_wells.items():
        #         print(f"{group}:")
        #         for row, cols in sorted(wells.items()):  # Sort rows alphabetically
        #             print(f"  {row}: {', '.join(map(str, cols))}")


    def save_geno_file(self, output_file):
        merged_data = []  # Use a list to store individual DataFrames

        for box_id, data in self.plates.items():
            temp = data[['Well Position', 'Genotype']].copy()  # Avoid SettingWithCopyWarning
            temp['box'] = box_id  # Add box identifier
            merged_data.append(temp)  # Append to list

        merged_data = pd.concat(merged_data, ignore_index=True)  # Combine all into a single DataFrame
        merged_data.columns = ['well', 'genotype', 'box']

        # Replace NaN values in genotype with "Excluded"
        # Use assignment instead of inplace modification to avoid FutureWarning
        merged_data['genotype'] = merged_data['genotype'].fillna('Excluded')

        # If there are controls, remove them
        df_filtered = merged_data[~merged_data.apply(lambda row: row["well"] in self.controls.get(row["box"], []), axis=1)]

        # Ensure output directory exists
        os.makedirs(output_file, exist_ok=True)

        # Save as a proper CSV
        df_filtered.to_csv(f"{output_file}/{output_file}_genotypes.csv", index=False)
        print(f"Genotype file saved as {output_file}/{output_file}_genotypes.csv")

        return df_filtered


    # def save_geno_file_SA(self, plate, output_file):
    #     """
    #     SINGLE PLATES
    #     FOR MATLAB Sleep Analysis.Generate genotype text files from the interpreted data.
    #
    #     Args:
    #         output_file (str): Path to save the generated genotype file.
    #     """
    #
    #     # Map well positions (e.g., A1 to 1, H12 to 96)
    #     well_to_numeric = {
    #         f"{row}{col}": idx
    #         for idx, (row, col) in enumerate(
    #             [(row, col) for row in 'ABCDEFGH' for col in range(1, 13)], start=1
    #         )
    #     }
    #
    #     data = self.plates[plate]
    #     # Map numeric well positions in the data
    #     data['Numeric Well'] = data['Well Position'].map(well_to_numeric)
    #
    #     # Group numeric wells by genotype
    #     grouped_wells = {
    #         genotype: data[data['Genotype'] == genotype]['Numeric Well'].tolist()
    #         for genotype in ['WT', 'HET', 'HOM']
    #     }
    #
    #     # Pad each list to the maximum length
    #     max_length = max(len(values) for values in grouped_wells.values())
    #     data_padded = {key: values + [None] * (max_length - len(values)) for key, values in grouped_wells.items()}
    #
    #     # Create a DataFrame
    #     df = pd.DataFrame(data_padded).fillna("")
    #
    #     # Convert all numbers to integers where applicable
    #     df = df.apply(lambda col: col.map(lambda x: int(x) if isinstance(x, float) or isinstance(x, int) else x))
    #
    #     # Define the new header
    #     new_header = ['genotype1', 'genotype1', 'genotype1']
    #     genotypes = ['WT', 'HET', 'HOM']
    #
    #     # Push the original header down by appending it as the first row
    #     df.columns = [genotypes[i] for i in range(len(df.columns))]  # Temporary column names
    #     df.loc[-1] = df.columns  # Add the original header as a row
    #     df.index = df.index + 1  # Shift index
    #     df = df.sort_index()  # Sort index to place the new row at the top
    #
    #     # Replace the header with the new header
    #     df.columns = new_header
    #
    #
    #     # Save the DataFrame to a .txt file with tab-separated values
    #     df.to_csv(f"{output_file}genotype.txt", index=False, sep='\t')
    #
    #     # Confirmation message
    #     print(f"DataFrame saved to {output_file}")

    def produce_geno_file(self, output_file, merged=False):

        well_to_numeric = {
            f"{row}{col}": idx
            for idx, (row, col) in enumerate(
                [(row, col) for row in 'ABCDEFGH' for col in range(1, 13)], start=1
            )
        }

        def _build_geno_df(df, offset=0):
            df = df.copy()
            df['Numeric Well'] = df['Well Position'].map(well_to_numeric) + offset
            grouped_wells = {
                geno: df[df['Genotype'] == geno]['Numeric Well'].tolist()
                for geno in ['WT', 'HET', 'HOM']
            }
            max_length = max(len(v) for v in grouped_wells.values()) if grouped_wells else 0
            padded = {k: v + [None] * (max_length - len(v)) for k, v in grouped_wells.items()}
            out = pd.DataFrame(padded).fillna("")
            out = out.apply(lambda col: col.map(lambda x: int(x) if isinstance(x, (float, int)) and x != 0 else x))
            # Push header down as first row
            out.columns = ['WT', 'HET', 'HOM']
            out.loc[-1] = out.columns
            out.index = out.index + 1
            out = out.sort_index()
            out.columns = ['genotype1', 'genotype1', 'genotype1']
            return out

        if merged:
            chunks = []
            for i, plate_idx in enumerate(sorted(self.plates.keys())):
                plate_df = self.plates[plate_idx].copy()
                plate_df['Numeric Well'] = plate_df['Well Position'].map(well_to_numeric) + i * 96
                chunks.append(plate_df)
            combined = pd.concat(chunks, ignore_index=True)
            # Use _build_geno_df logic but with pre-computed numeric wells
            grouped_wells = {
                geno: combined[combined['Genotype'] == geno]['Numeric Well'].tolist()
                for geno in ['WT', 'HET', 'HOM']
            }
            max_length = max(len(v) for v in grouped_wells.values()) if grouped_wells else 0
            padded = {k: v + [None] * (max_length - len(v)) for k, v in grouped_wells.items()}
            df = pd.DataFrame(padded).fillna("")
            df = df.apply(lambda col: col.map(lambda x: int(x) if isinstance(x, (float, int)) and x != 0 else x))
            df.columns = ['WT', 'HET', 'HOM']
            df.loc[-1] = df.columns
            df.index = df.index + 1
            df = df.sort_index()
            df.columns = ['genotype1', 'genotype1', 'genotype1']
            output_path = f"{output_file}genotype.txt"
            df.to_csv(output_path, index=False, sep='\t')
            print(f"Merged genotype file saved to {output_path}")
        else:
            for plate_idx in sorted(self.plates.keys()):
                df = _build_geno_df(self.plates[plate_idx])
                output_path = f"{output_file}_plate{plate_idx}_genotype.txt"
                df.to_csv(output_path, index=False, sep='\t')
                print(f"Plate {plate_idx} genotype file saved to {output_path}")

    def _well_index_to_position(self, index, cols=12):
        """Convert 0-based well index to position string, e.g. 0 -> 'A1', 13 -> 'B2'."""
        row = index // cols
        col = index % cols + 1
        return f"{chr(65 + row)}{col}"


    def read_eds(self, filepath):
        """
        Read a QuantStudio .eds file and return a DataFrame matching
        the Genotyping Results CSV export.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to the .eds file.

        Returns
        -------
        pd.DataFrame
            Columns: Well, Well Position, Sample, Allele 1, Allele 2, Call
            - Well: 1-based well number (int)
            - Well Position: e.g. 'A1', 'H12' (str)
            - Sample: sample name from plate setup (str)
            - Allele 1: FAM Rn = FAM_signal / ROX_signal (float)
            - Allele 2: VIC Rn = VIC_signal / ROX_signal (float)
            - Call: empty string (calls are computed by TF software, not stored in .eds)
        """
        with zipfile.ZipFile(filepath, "r") as zf:
            # ── 1. Plate setup: well → sample name ──────────────────────
            with zf.open("setup/plate_setup.json") as f:
                plate_setup = json.load(f)

            well_sample = {}
            for w in plate_setup.get("wells", []):
                well_sample[w["index"]] = w.get("sampleName", "")

            # Determine allele-to-dye mapping from SNP assay definition
            snp_assays = plate_setup.get("snpAssays", [])
            if snp_assays:
                allele1_reporter = snp_assays[0]["allele1"]["reporter"]  # e.g. "FAM"
                allele2_reporter = snp_assays[0]["allele2"]["reporter"]  # e.g. "VIC"
            else:
                allele1_reporter = "FAM"
                allele2_reporter = "VIC"

            passive_ref = plate_setup.get("passiveReference", "ROX")

            # ── 2. Multicomponent data: raw fluorescence signals ────────
            with zf.open("apldbio/sds/multicomponentdata.xml") as f:
                mc_tree = ET.parse(f)
            mc_root = mc_tree.getroot()

            well_count = int(mc_root.findtext("WellCount", "96"))

            # Build a per-well dye order lookup.
            # Most files have identical dye lists per well, but we check
            # each well in case they differ.
            well_dye_orders = {}
            for dd in mc_root.findall(".//DyeData"):
                widx = int(dd.get("WellIndex"))
                dye_text = dd.find("DyeList").text  # e.g. "[FAM,ROX,VIC]"
                dye_order = [d.strip() for d in dye_text.strip("[]").split(",")]
                well_dye_orders[widx] = dye_order

            # ── 3. Extract Rn values per well ───────────────────────────
            rows = []

            for well_idx in range(well_count):
                signal = mc_root.find(f".//SignalData[@WellIndex='{well_idx}']")
                if signal is None:
                    # Well has no signal data — include with NaN values
                    rows.append({
                        "Well": well_idx + 1,
                        "Well Position": _well_index_to_position(well_idx),
                        "Sample": well_sample.get(well_idx, ""),
                        "Allele 1": float("nan"),
                        "Allele 2": float("nan"),
                        "Call": "",
                    })
                    continue

                cycle_data = signal.findall("CycleData")
                dye_order = well_dye_orders.get(well_idx)

                if dye_order is None or len(cycle_data) == 0:
                    rows.append({
                        "Well": well_idx + 1,
                        "Well Position": self._well_index_to_position(well_idx),
                        "Sample": well_sample.get(well_idx, ""),
                        "Allele 1": float("nan"),
                        "Allele 2": float("nan"),
                        "Call": "",
                    })
                    continue

                # Validate that we have enough CycleData for all dyes
                if len(cycle_data) != len(dye_order):
                    raise ValueError(
                        f"Well {well_idx} ({_well_index_to_position(well_idx)}): "
                        f"expected {len(dye_order)} CycleData entries for dyes "
                        f"{dye_order}, but found {len(cycle_data)}. "
                        f"Use debug_eds() to inspect this file."
                    )

                # Build dye_name -> signal mapping
                dye_signals = {}
                for dye_name, cd in zip(dye_order, cycle_data):
                    vals = [float(v) for v in cd.text.strip("[]").split(",")]
                    dye_signals[dye_name] = vals[-1]  # last step value

                ref_signal = dye_signals.get(passive_ref, 0.0)
                allele1_signal = dye_signals.get(allele1_reporter, 0.0)
                allele2_signal = dye_signals.get(allele2_reporter, 0.0)

                # Rn = dye_signal / passive_reference_signal
                allele1_rn = allele1_signal / ref_signal if ref_signal != 0 else 0.0
                allele2_rn = allele2_signal / ref_signal if ref_signal != 0 else 0.0

                rows.append({
                    "Well": well_idx + 1,  # 1-based to match CSV
                    "Well Position": self._well_index_to_position(well_idx),
                    "Sample": well_sample.get(well_idx, ""),
                    "Allele 1": allele1_rn,
                    "Allele 2": allele2_rn,
                    "Call": "",
                })

        return pd.DataFrame(rows)

import subprocess
import os

class Video():

    def __init__(self, base_name):
        self.base_name = base_name

    def convert_avi_to_mp4(self):

        input_avi = f"{base_name}/{base_name}_Box1_0001.avi"
        converted_mp4 = f"{base_name}/{base_name}_Box1_0001_converted.mp4"
        output_dir = f"{base_name}/video_segments"
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Convert full AVI to MP4
        if not os.path.exists(converted_mp4):
            print("🎬 Converting full AVI to MP4...")
            conversion_command = [
                "ffmpeg",
                "-i", input_avi,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-movflags", "+faststart",
                "-y",  # Overwrite if exists
                converted_mp4
            ]
            subprocess.run(conversion_command, check=True)
        else:
            print("✅ MP4 conversion already exists. Skipping.")

    def overlay_genotype(self):

        return None
