# Standard library imports
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from concurrent.futures import ThreadPoolExecutor
import dask.dataframe as dd
from dask.distributed import Client


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
        temp_df['stdate_sttime'] = pd.to_datetime(temp_df['stdate'] + ' ' + temp_df['sttime'], dayfirst=True)
        print(f"Start of experiment: {temp_df['stdate_sttime'][0]}")

        return temp_df['stdate_sttime'][0]

    def get_genotype_df(self):
        # geno = self.find_map_files(self.path)
        # self.geno_map_files = self.find_map_files(self.path)['Genotype Map Files'][0]
        # self.cond_map_files = self.find_map_files(self.path)['Condition Map Files'][0]

        df_list = []

        for i, geno_file in enumerate(self.geno_map_files):

            current_box = re.search(r'_(\d+)_genotypeMap', geno_file).group(1)


            # Tidy genotype and condMap data, adding box identifiers
            genotype_data = pd.read_excel(self.path+geno_file) # If the Excel file is open, this might throw an engine error
            geno_df = self.strip_96well_data(genotype_data, current_box, 'genotype')

            df_list.append(geno_df)

        genotype_df = pd.concat(df_list, ignore_index=True)

        print(genotype_df)

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

        # Call the parent class's __init__
        super().__init__(date, box1, box2, exp, export, omit=omit)

        # if not os.path.exists(f"{self.path}{self.name}_raw_df.csv"):
        #     self.df = self.prepare_raw_data()
        self.df = self.prepare_raw_data()



        if self.omit is None:
            return None
        else:
            return None

    def prepare_raw_data(self):
        mega_df_file = f"{self.path}{self.name}_raw_df.csv"
        cols         = ["abstime", "time", "type", "location", "data1"]

        if not os.path.exists(mega_df_file):
            self.combine_csv_files_dask()
            raise FileNotFoundError("The file does not exist. You need to combine the csv files first. obj.combine_csv_files()")

        print(f"Preparing {mega_df_file}.")
        dirty_data = pd.read_csv(mega_df_file, usecols=cols)



        # Filter rows and create a copy
        filtered_df = dirty_data[dirty_data['type'] == 101].copy()

        # ADJUST TIME
        # Vectorized operations for creating new columns
        time_in_seconds = filtered_df["time"] / 1_000_000
        filtered_df.loc[:, "fullts"] = self.correct_start_time + pd.to_timedelta(time_in_seconds, unit="s")
        filtered_df.loc[:, "zhrs"] = (filtered_df["fullts"] - pd.Timestamp(self.zt0)).dt.total_seconds() / 3600
        filtered_df.loc[:, "exsecs"] = time_in_seconds


        # Move the new columns to the front
        columns_order = ["fullts", "zhrs", "exsecs"] + [col for col in filtered_df.columns if col not in ["fullts", "zhrs", "exsecs"]]
        filtered_df = filtered_df[columns_order]

        # Drop old columns (optional)
        # filtered_df.drop(columns=["abstime", "time", "type"], inplace=True)

        genotype_df = self.get_genotype_df()
        # condition_df = self.get_condition_df()
        # print(geno_df, cond_df)

        filtered_df = self.convert_location_column(filtered_df)


        if self.cond_map_files:
            condition_df = self.get_condition_df()
            # Merge data on Location and Box
            merged_data = filtered_df.merge(genotype_df, on=['box', 'well'], how='left').merge(condition_df, on=['box', 'well'], how='left')
        else:
            merged_data = filtered_df.merge(genotype_df, on=['box', 'well'], how='left')

        # # Set WT as default genotype where missing
        # merged_data['genotype'].fillna('WT', inplace=True)
        merged_data['genotype'].dropna(inplace=True)

        merged_data = merged_data[~merged_data['genotype'].isin(['empty', 'NaN'])]

        print('Done')
        return merged_data

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


    # def combine_csv_files_dask(self, output_file=None):
    #     # Define input and output paths
    #     csv_folder = f"{self.path}{self.name}_rawoutput/raw_converted_csv/"
    #     output_file = output_file or f"{self.path}{self.name}_raw_df.csv"
    #
    #     # Define consistent dtypes for all columns
    #     dtype = {
    #         'abstime': 'object',  # Will be parsed later as datetime
    #         'time': 'object',     # Assuming it's a string
    #         'channel': 'float64', # Ensuring consistent float type
    #         'type': 'float64',    # Ensuring consistent float type
    #         'location': 'object', # Assuming it's a string
    #         'data1': 'float64'    # Assuming numeric
    #     }
    #
    #     # Use a pattern to load all CSV files
    #     csv_files_pattern = os.path.join(csv_folder, "*.csv")
    #
    #     # Load all CSV files into a Dask DataFrame
    #     ddf = dd.read_csv(
    #         csv_files_pattern,
    #         usecols=['abstime', 'time', 'channel', 'type', 'location', 'data1'],
    #         dtype=dtype
    #     )
    #
    #     # Safely handle abstime: prioritize numeric parsing for 'unit'
    #     ddf['abstime'] = dd.to_datetime(
    #         dd.to_numeric(ddf['abstime'], errors='coerce'),
    #         unit='ms', errors='coerce'
    #     ).fillna(
    #         dd.to_datetime(ddf['abstime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    #     )
    #
    #     # Drop rows with invalid abstime
    #     ddf = ddf[ddf['abstime'].notnull()]
    #
    #     # Compute and optionally save to CSV
    #     self.mega_dataframe = ddf.compute()
    #     if output_file:
    #         self.mega_dataframe.to_csv(output_file, index=False)
    #     return self.mega_dataframe

    def combine_csv_files_dask(self, output_file=None):
        # Define input and output paths
        csv_folder = f"{self.path}{self.name}_rawoutput/raw_converted_csv/"
        output_file = output_file or f"{self.path}{self.name}_raw_df.csv"

        #**Limit resources: ≤6 cores, ≤30GB memory**
        client = Client(n_workers=6, threads_per_worker=1, memory_limit='5GB')  # 6 workers x 5GB each = 30GB

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

        # **Do not compute everything into memory**
        # Instead of `compute()`, write to CSV using Dask
        if output_file:
            ddf.to_csv(output_file, index=False, single_file=True)  # Save efficiently

        # **Shut down Dask client to free resources**
        client.close()

        return ddf

    # def combine_csv_files(self, output_file=None):
    # # def combine_csv_files(self, input_folder, output_file=None):
    #     """
    #     Combines all CSV files in a folder into a single DataFrame.
    #     You first need to use the bash script (found in Scripts) in Terminal to batch convert xls to csv with SSCONVERT.
    #     e.g. ./batch_convert_xls_to_csv.sh 241107_16_17_PNPO_PTZ/241107_16_17_PNPO_PTZ_rawoutput
    #     This method will not work if there is an existing combined csv inside the input folder (csv_folder).
    #
    #     Args:
    #         input_folder (str): Path to the folder containing CSV files.
    #         output_file (str, optional): Path to save the combined DataFrame as a CSV.
    #
    #     Returns:
    #         pandas.DataFrame: Combined DataFrame of all CSV files.
    #     """
    #     csv_folder  = f"{self.path}{self.name}_rawoutput/raw_converted_csv/"
    #     output_file = f"{self.path}{self.name}_raw_df.csv"
    #
    #     # List and sort all CSV files in the input folder by numeric order
    #     csv_files = sorted(
    #         [f for f in os.listdir(csv_folder) if f.endswith('.csv')],
    #         key=lambda x: int(re.search(r'_(\d+)\.csv$', x).group(1))  # Match digits at the end
    #     )
    #
    #     if not csv_files:
    #         print(f"No CSV files found in {csv_folder}.")
    #         return None
    #
    #     cols = ['abstime', 'time', 'channel', 'type', 'location', 'data1']
    #
    #     # Initialize an empty list to store DataFrames
    #     dataframes = []
    #
    #     # Iterate through each CSV file
    #     for csv_file in csv_files:
    #         file_path = os.path.join(csv_folder, csv_file)
    #         print(f"Reading {file_path}")
    #         df = pd.read_csv(file_path, usecols=cols, parse_dates=['abstime'])  # Adjust for delimiter if necessary
    #         dataframes.append(df)
    #
    #     # Combine all DataFrames into one
    #     self.mega_dataframe = pd.concat(dataframes, ignore_index=True)
    #     print("All files have been combined into a single DataFrame.")
    #
    #     # Save the combined DataFrame as a CSV if specified
    #     if output_file:
    #         self.mega_dataframe.to_csv(output_file, index=False)
    #         print(f"Combined DataFrame saved to {output_file}")
    #
    #     return self.mega_dataframe


class MiddurData(Experiment): #The output is not compatible with sleep analysis
    # def __init__(self, date, box1, box2, exp, export=False):
    def __init__(self, date, box1, box2, exp, export=False, nbox=2, cbox=None, omit=None):

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

        #
        # print(merged_data['condition'].unique())
        # empty = merged_data['genotype'].isin(['empty', 'excluded'])
        # print(merged_data.iloc[4411],empty)

        # # Remove rows where 'genotype' is 'empty', None, or an empty string
        # # merged_data = merged_data[~merged_data['genotype'].isin(['empty', 'excluded', None, '']) & ~merged_data['condition'].isin(['empty', 'excluded', None, ''])]
        # merged_data = merged_data[~merged_data['genotype'].isin(['empty', 'excluded', 'NaN'])]
        merged_data = merged_data[~merged_data['genotype'].isin(['empty', 'NaN'])]
        #
        # # Drop rows where Genotype or Treatment is explicitly marked as "empty"
        # prepped_data = merged_data[(merged_data['genotype'] != 'empty') & (merged_data['condition'] != 'empty')]
        # prepped_data = merged_data[(merged_data['genotype'] != 'empty')]

        prepped_filtered_data = self.filter_df_by_omit(merged_data, self.omit)

        # # Organize columns
        prepped_complete_data = merged_data
        #
        # # Calculate the 'CLOCK' column based on the combined 'stdate' and 'sttime'
        # prepped_data['clock'] = self.calculate_clock(prepped_data['stdate_sttime'])
        print('Finished prepping data.')

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
        box_data = data[data['box'] == box_id].copy()

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
        filename = os.path.join(path, f"{date}_00_DATA.csv")  # Construct the file path
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

    def __init__(self, csv_files, omitted_wells=None, drop_wells=None, display_list=False):
        self.csv_files     = csv_files
        self.omitted_wells = {box: set(wells) for box, wells in (omitted_wells or {}).items()}
        self.drop_wells    = {box: set(wells) for box, wells in (drop_wells or {}).items()}
        self.display_list  = display_list

        # Dictionary to store processed data for each plate
        self.plates = {}

        # Load and process each file
        for box_id, file_path in self.csv_files.items():
            self.plates[box_id] = self.load_data(box_id)
            self.plot_allele_and_well_plate(box_id)
            if display_list:
                self.print_grouped_well_lists(box_id, self.plates[box_id])


    def load_data(self, box_id):
        """Loads and preprocesses the data from the CSV file."""
        data = pd.read_csv(self.csv_files[box_id], skiprows=23,
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
        self.perform_clustering(data, box_id)
        file_name = Path(self.csv_files[box_id]).stem
        print(file_name)
        updated_colors = {'WT': '#ff7f0e', 'HET': '#2ca02c', 'HOM': '#1f77b4'}
        data['Color'] = data['Genotype'].map(updated_colors)

        # print(self.data[['Well Position','Call','Cluster','Genotype','Color']])

        # Set up the figure with two subplots (for Allele plot and 96-well visualization)
        fig, axs = plt.subplots(1, 2, figsize=(20, 6))

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
        plt.show()

    def get_grouped_well_lists(self, box_id, data):
        """
        Groups wells into WT, HET, HOM, Omitted, and Dropped categories using the object's attributes.
        Returns the lists as formatted dictionaries.
        """
        all_wells = set(data['Well Position'])  # All wells in the dataset
        omitted_set = set(self.omitted_wells[box_id]) if self.omitted_wells[box_id] else set()
        dropped_set = set(self.drop_wells[box_id]) if self.drop_wells[box_id] else set()

        # Exclude omitted and dropped wells from WT, HET, HOM
        valid_wells = all_wells - omitted_set - dropped_set
        group_wells = {
            'WT': data[(data['Genotype'] == 'WT') & (data['Well Position'].isin(valid_wells))]['Well Position'].tolist(),
            'HET': data[(data['Genotype'] == 'HET') & (data['Well Position'].isin(valid_wells))]['Well Position'].tolist(),
            'HOM': data[(data['Genotype'] == 'HOM') & (data['Well Position'].isin(valid_wells))]['Well Position'].tolist(),
            'Omitted': list(omitted_set & all_wells),
            'Dropped': list(dropped_set),
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


    def produce_geno_file(self, output_file):
        """
        FOR MATLAB Sleep Analysis.Generate genotype text files from the interpreted data.

        Args:
            output_file (str): Path to save the generated genotype file.
        """

        # Map well positions (e.g., A1 to 1, H12 to 96)
        well_to_numeric = {
            f"{row}{col}": idx
            for idx, (row, col) in enumerate(
                [(row, col) for row in 'ABCDEFGH' for col in range(1, 13)], start=1
            )
        }

        # Map numeric well positions in the data
        self.data['Numeric Well'] = self.data['Well Position'].map(well_to_numeric)

        # Group numeric wells by genotype
        grouped_wells = {
            genotype: self.data[self.data['Genotype'] == genotype]['Numeric Well'].tolist()
            for genotype in ['WT', 'HET', 'HOM']
        }

        # Pad each list to the maximum length
        max_length = max(len(values) for values in grouped_wells.values())
        data_padded = {key: values + [None] * (max_length - len(values)) for key, values in grouped_wells.items()}

        # Create a DataFrame
        df = pd.DataFrame(data_padded).fillna("")

        # Convert all numbers to integers where applicable
        df = df.apply(lambda col: col.map(lambda x: int(x) if isinstance(x, float) or isinstance(x, int) else x))

        # Define the new header
        new_header = ['genotype1', 'genotype1', 'genotype1']
        genotypes = ['WT', 'HET', 'HOM']

        # Push the original header down by appending it as the first row
        df.columns = [genotypes[i] for i in range(len(df.columns))]  # Temporary column names
        df.loc[-1] = df.columns  # Add the original header as a row
        df.index = df.index + 1  # Shift index
        df = df.sort_index()  # Sort index to place the new row at the top

        # Replace the header with the new header
        df.columns = new_header


        # Save the DataFrame to a .txt file with tab-separated values
        df.to_csv(f"{output_file}genotype.txt", index=False, sep='\t')

        # Confirmation message
        print(f"DataFrame saved to {output_file}")
