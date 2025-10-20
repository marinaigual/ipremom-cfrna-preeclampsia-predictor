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


# ------------------- QUALITY READS-----------------
def extract_quality_scores(zip_path: str) -> float:
    """
    Extract quality scores and counts from fastqc_data.txt inside a ZIP file.
    
    Args:
        zip_path (str): Path to the .zip file containing fastqc_data.txt.
        
    Returns:
        float: Average quality score for the sample.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if "fastqc_data.txt" in file:
                with zip_ref.open(file) as f:
                    lines = f.read().decode('utf-8').splitlines()
                    # Locate the "Per sequence quality scores" section
                    start = lines.index(">>Per sequence quality scores	pass") + 1
                    end = lines.index(">>END_MODULE", start)
                    quality_data = lines[start:end]
                    
                    quality_scores = []
                    counts = []
                    for line in quality_data:
                        if line.startswith("#") or line.startswith(">>"):
                            continue  # Skip headers or metadata
                        quality, count = map(float, line.split())
                        quality_scores.append(quality)
                        counts.append(count)
                    
                    # Calculate the average
                    total_count = sum(counts)
                    average_quality = sum(q * c for q, c in zip(quality_scores, counts)) / total_count
                    return average_quality
    return None

def calculate_paired_quality(r1_zip: str, r2_zip: str) -> float:
    """
    Calculate average quality score for paired-end reads.
    
    Args:
        r1_zip (str): Path to the R1 .zip file.
        r2_zip (str): Path to the R2 .zip file.
        
    Returns:
        float: Average quality score for the paired-end sample.
    """
    quality_r1 = extract_quality_scores(r1_zip)
    quality_r2 = extract_quality_scores(r2_zip)
    if quality_r1 is not None and quality_r2 is not None:
        return (quality_r1 + quality_r2) / 2
    return None

def process_run_quality_reads(run_dir: str) -> pd.DataFrame:
    """
    Process a single run directory to calculate average quality scores for all samples.
    
    Args:
        run_dir (str): Path to the directory containing fastqc.zip files for a single run.
        
    Returns:
        pd.DataFrame: DataFrame with sample names and average quality scores.
    """
    samples = {}
    zip_files = [f for f in os.listdir(run_dir) if f.endswith(".zip")]
    
    for file in zip_files:
        # Extract base sample name (ignoring R1/R2 distinction and other suffixes)
        sample_base = re.sub(r'_R[12]_.*', '', file)
        if sample_base not in samples:
            samples[sample_base] = {"R1": None, "R2": None}
        if "_R1_" in file:
            samples[sample_base]["R1"] = os.path.join(run_dir, file)
        elif "_R2_" in file:
            samples[sample_base]["R2"] = os.path.join(run_dir, file)
    
    # Calculate quality scores for all paired samples
    results = []
    for sample, paths in samples.items():
        if paths["R1"] and paths["R2"]:  # Ensure both R1 and R2 exist
            avg_quality = calculate_paired_quality(paths["R1"], paths["R2"])
            results.append({"Sample": sample, "Mean R1-R2 Quality": avg_quality})
    
    return pd.DataFrame(results)

def aggregate_runs_quality_reads(base_dir, process_all=True, specific_run=None):
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
        run_coverage = process_run_quality_reads(run_dir)
        all_coverage_results.append(run_coverage)

    # Combine results from all runs
    combined_df = pd.concat(all_coverage_results, ignore_index=True)

    # Extract base sample name and calculate mean coverage
    combined_df['Sample Base'] = combined_df['Sample'].str.extract(r'(.*)_L00\d')
    grouped_data = combined_df.groupby('Sample Base', as_index=False).agg({
        'Mean R1-R2 Quality': 'mean',
        'Sample': lambda x: ', '.join(x)
    })
    grouped_data.rename(columns={'Mean R1-R2 Quality': 'Average Mean Quality'}, inplace=True)
    grouped_data['Sample Base'] = grouped_data['Sample'].str.extract(r'^(.*?)_')

    return grouped_data

    # return combined_df






