import os
import re
import zipfile
import pandas as pd
from tqdm import tqdm
import yaml

# Load configuration from YAML
with open("../../../config/RNAseq/RNAseq_config.yml", "r") as file:
        config = yaml.safe_load(file)

TOTAL_TRANSCRIPTOME_LENGTH = config['TOTAL_TRANSCRIPTOME_LENGTH'][0]


# ------------------- NUMBER READS -----------------
def extract_fastqc_data(zip_path):
    """
    Extracts Total Sequences and Sequence Length from the 'fastqc_data.txt' file inside a ZIP archive.
    
    Args:
        zip_path (str): Path to the fastqc ZIP file.
    
    Returns:
        tuple: A tuple containing:
            - total_sequences (int): Total number of sequences.
            - sequence_length (float): Average sequence length.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if "fastqc_data.txt" in file:
                with zip_ref.open(file) as f:
                    lines = f.read().decode('utf-8').splitlines()
                    total_sequences = None
                    sequence_length = None
                    for line in lines:
                        if line.startswith("Total Sequences"):
                            total_sequences = int(line.split()[-1])
                        elif line.startswith("Sequence length"):
                            seq_length_str = line.split()[-1]
                            if '-' in seq_length_str:
                                sequence_length = sum(map(int, seq_length_str.split('-'))) / 2
                            else:
                                sequence_length = int(seq_length_str)
                    return total_sequences, sequence_length
    return None, None

def calculate_number_reads(total_sequences, seq_length):
    """
    Calculates the coverage for a single sample.
    
    Args:
        total_sequences (int): Total number of sequences.
        seq_length (float): Average sequence length.
    
    Returns:
        float: Coverage value for the sample.
    """
    return round(total_sequences/seq_length, 0)

def process_run_number_reads(run_path):
    """
    Processes a single run directory, extracting coverage metrics for all samples.
    
    Args:
        run_path (str): Path to the directory containing the fastqc ZIP files.
    
    Returns:
        pd.DataFrame: A DataFrame containing coverage metrics for all samples in the run.
    """
    samples = {}

    # List all fastqc ZIP files in the run directory
    list_samples = [x for x in os.listdir(run_path) if "R1" in x and x.endswith(".zip")]

    for file in list_samples:
        sample_name = re.sub(r'_R1_.*', '', file)
        zip_path = os.path.join(run_path, file)
        total_sequences, seq_length = extract_fastqc_data(zip_path)
        if sample_name not in samples:
            samples[sample_name] = []
        samples[sample_name].append((total_sequences, seq_length))

    # Calculate coverage for each sample
    coverage_results = []
    for sample, metrics in samples.items():
        coverages = []
        for total_sequences, seq_length in metrics:
            if total_sequences is not None and seq_length is not None:
                coverages.append(calculate_number_reads(total_sequences, seq_length))
        if coverages:
            mean_coverage = sum(coverages) / len(coverages)
            coverage_results.append({"Sample": sample, "Mean Number Reads": mean_coverage})

    # Convert results into a DataFrame
    coverage_df = pd.DataFrame(coverage_results)
    return coverage_df

def aggregate_runs_number_reads(base_dir, process_all=True, specific_run=None):
    """
    Processes multiple runs or a single run, aggregates the coverage data, and calculates the mean coverage.
    
    Args:
        base_dir (str): Base directory containing the run directories.
        process_all (bool): Whether to process all runs. If False, processes only `specific_run`.
        specific_run (str): Name of the specific run to process. Ignored if `process_all` is True.
    
    Returns:
        pd.DataFrame: A DataFrame containing the aggregated coverage data for all processed runs.
    """
    run_dirs = [os.path.join(base_dir, run) for run in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, run))]

    if not process_all:
        if specific_run:
            run_dirs = [os.path.join(base_dir, specific_run)]
        else:
            raise ValueError("`specific_run` must be provided if `process_all` is False.")
    
    all_coverage_results = []

    # Iterate through runs
    for run_dir in tqdm(run_dirs, desc="Processing runs"):
        run_coverage = process_run_number_reads(run_dir)
        all_coverage_results.append(run_coverage)

    # Combine results from all runs
    combined_df = pd.concat(all_coverage_results, ignore_index=True)

    # Extract base sample name and calculate mean coverage
    combined_df['Sample Base'] = combined_df['Sample'].str.extract(r'(.*)_L00\d')
    grouped_data = combined_df.groupby('Sample Base', as_index=False).agg({
        'Mean Number Reads': 'mean',
        'Sample': lambda x: ', '.join(x)
    })
    grouped_data.rename(columns={'Mean Number Reads': 'Average Mean Number Reads'}, inplace=True)
    grouped_data['Sample Base'] = grouped_data['Sample'].str.extract(r'^(.*?)_')

    return grouped_data





