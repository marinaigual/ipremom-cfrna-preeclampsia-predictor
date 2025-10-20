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


# ------------------- SEQUENCE DEPTH -----------------

def extract_fastqc_sequencing_data(zip_path: str) -> tuple:
    """
    Extract Total Sequences and Deduplicated Percentage from fastqc_data.txt inside a ZIP file.
    
    Args:
        zip_path (str): Path to the fastqc.zip file.
        
    Returns:
        tuple: (Total Sequences, Deduplicated Percentage)
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if "fastqc_data.txt" in file:
                with zip_ref.open(file) as f:
                    lines = f.read().decode('utf-8').splitlines()
                    total_sequences = None
                    dedup_percentage = None
                    for line in lines:
                        if line.startswith("Total Sequences"):
                            total_sequences = int(line.split()[-1])
                        elif line.startswith("#Total Deduplicated Percentage"):
                            dedup_percentage = float(line.split()[-1]) / 100.0  # Convert to fraction
                    return total_sequences, dedup_percentage
    return None, None

def calculate_sequencing_depth(total_sequences: int, dedup_percentage: float) -> float:
    """
    Calculate sequencing depth based on Total Sequences and Deduplicated Percentage.
    
    Args:
        total_sequences (int): Total number of sequences.
        dedup_percentage (float): Deduplicated percentage as a fraction (e.g., 0.5 for 50%).
        
    Returns:
        float: Calculated sequencing depth.
    """
    if total_sequences is not None and dedup_percentage is not None:
        return total_sequences * dedup_percentage
    return None

def calculate_paired_depth(r1_path: str, r2_path: str) -> float:
    """
    Calculate the average sequencing depth for paired R1 and R2 files.
    
    Args:
        r1_path (str): Path to the R1 fastqc.zip file.
        r2_path (str): Path to the R2 fastqc.zip file.
        
    Returns:
        float: Average sequencing depth for the paired files.
    """
    total_sequences_r1, dedup_percentage_r1 = extract_fastqc_sequencing_data(r1_path)
    depth_r1 = calculate_sequencing_depth(total_sequences_r1, dedup_percentage_r1)
    
    total_sequences_r2, dedup_percentage_r2 = extract_fastqc_sequencing_data(r2_path)
    depth_r2 = calculate_sequencing_depth(total_sequences_r2, dedup_percentage_r2)
    
    if depth_r1 is not None and depth_r2 is not None:
        return (depth_r1 + depth_r2) / 2
    return None

def process_run_sequencing_depth(run_dir: str) -> pd.DataFrame:
    """
    Process a single run directory to calculate sequencing depth for all samples.
    
    Args:
        run_dir (str): Path to the directory containing fastqc.zip files for a single run.
        
    Returns:
        pd.DataFrame: DataFrame with sample names and average sequencing depths.
    """
    samples = {}
    zip_files = [f for f in os.listdir(run_dir) if f.endswith(".zip")]
    
    for file in zip_files:
        sample_base = re.sub(r'_R[12]_.*', '', file)  # Extract base sample name
        if sample_base not in samples:
            samples[sample_base] = {"R1": None, "R2": None}
        if "_R1_" in file:
            samples[sample_base]["R1"] = os.path.join(run_dir, file)
        elif "_R2_" in file:
            samples[sample_base]["R2"] = os.path.join(run_dir, file)
    
    results = []
    for sample, paths in samples.items():
        if paths["R1"] and paths["R2"]:
            avg_depth = calculate_paired_depth(paths["R1"], paths["R2"])
            results.append({"Sample": sample, "Mean R1-R2 Sequencing Depth": avg_depth})
    
    return pd.DataFrame(results)

def aggregate_runs_sequencing_depth(base_dir: str, all_runs: bool = True, run_dir: str = None) -> pd.DataFrame:
    """
    Process multiple runs or a single run to calculate sequencing depth and aggregate results.
    
    Args:
        base_dir (str): Base directory containing multiple run directories.
        all_runs (bool): Whether to process all runs or a single specified run.
        run_dir (str): Specific run directory to process (used only if all_runs is False).
        
    Returns:
        pd.DataFrame: Aggregated DataFrame with sequencing depths for all runs/samples.
    """
    if not all_runs and run_dir is None:
        raise ValueError("If all_runs is False, a specific run_dir must be provided.")
    
    aggregated_results = []
    if all_runs:
        run_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    else:
        run_dirs = [run_dir]
    
    for run_path in tqdm(run_dirs, desc="Processing Runs"):
        run_results = process_run_sequencing_depth(run_path)
        aggregated_results.append(run_results)
    
    combined_df = pd.concat(aggregated_results, ignore_index=True)
    
    # Extract base sample name and calculate mean sequencing depth
    combined_df['Sample Base'] = combined_df['Sample'].str.extract(r'(.*)_L00\d')
    grouped_data = combined_df.groupby('Sample Base', as_index=False).agg({
        'Mean R1-R2 Sequencing Depth': 'mean',
        'Sample': lambda x: ', '.join(x)
    })
    grouped_data.rename(columns={'Mean R1-R2 Sequencing Depth': 'Average Mean Sequencing Depth'}, inplace=True)
    grouped_data['Sample Base'] = grouped_data['Sample'].str.extract(r'^(.*?)_')
    
    return grouped_data


