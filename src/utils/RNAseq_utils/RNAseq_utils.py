import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_total_transcript_length(gtf_file):
    # load GTF file
    col_names = ["chromosome", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
    gtf_data = pd.read_csv(gtf_file, sep="\t", comment="#", header=None, names=col_names)
    
    # filter rows with "transcript"
    transcripts = gtf_data[gtf_data["feature"] == "transcript"]
    
    # compute lenght of each transcript
    transcripts["length"] = transcripts["end"] - transcripts["start"] + 1
    
    # sum all lengths
    total_length = transcripts["length"].sum()
    
    print(f"Total transcript length: {total_length} bp")
    return total_length

def assign_group(dataframe: pd.DataFrame, 
                 lope_samples: list, 
                 eope_samples: list, 
                 control_samples: list) -> pd.DataFrame:
    """
    Assign a group (LOPE, EOPE, Control) to each sample base in the DataFrame.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame containing sample data.
        lope_samples (list): List of LOPE sample identifiers.
        eope_samples (list): List of EOPE sample identifiers.
        control_samples (list): List of Control sample identifiers.
    
    Returns:
        pd.DataFrame: Updated DataFrame with an additional 'Group' column.
    """
    def determine_group(sample_base: str) -> str:
        # extract the prefix (e.g., C02-P0013) from the sample base
        sample_prefix = re.sub(r'-T\d+.*', '', sample_base)
        if sample_prefix in lope_samples:
            return "LOPE"
        elif sample_prefix in eope_samples:
            return "EOPE"
        elif sample_prefix in control_samples:
            return "Control"
        else:
            return "Unknown"

    # apply the group assignment logic to each row in the DataFrame
    dataframe['Group'] = dataframe['Sample Base'].apply(determine_group)
    dataframe = dataframe[dataframe.Group != "Unknown"]
    return dataframe

def plot_results(
        dataframe: pd.DataFrame, 
        feature: str, 
        palette: dict = {'EOPE': '#FF9999', 'Control': '#9999FF', 'LOPE': '#FFD700'}
    ) -> None:
    """
    Plot the results for a given feature across timepoints and groups.

    Args:
        dataframe (pd.DataFrame): Input DataFrame containing the data to plot.
        feature (str): Feature to visualize (e.g., 'Coverage').
        palette (dict): Color mapping for groups (default includes EOPE, Control, LOPE).

    Returns:
        None
    """
    # Extract the timepoint (e.g., T1, T2, T3) from 'Sample Base' and add it as a new column
    dataframe['Timepoint'] = dataframe['Sample Base'].str.extract(r'-(T\d)')

    # Initialize the figure
    plt.figure(figsize=(12, 8))

    # Create a boxplot to display data distribution for each timepoint and group
    sns.boxplot(
        x='Timepoint', 
        y=f'Average Mean {feature}', 
        hue='Group', 
        data=dataframe, 
        palette=palette, 
        dodge=True,
        showfliers=False
    )

    # Overlay a strip plot to show individual data points
    sns.stripplot(
        x='Timepoint',
        y=f'Average Mean {feature}',
        hue='Group',
        data=dataframe,
        dodge=True,
        palette=palette,
        jitter=True,
        alpha=0.6,
        marker='o',
        linewidth=0.5,
        edgecolor='gray'
    )

    # Remove duplicate legend entries caused by the strip plot
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:len(palette)], labels[:len(palette)], title='Group', loc='upper right')

    # Identify and annotate outliers
    for timepoint_idx, timepoint in enumerate(dataframe['Timepoint'].unique()):
        for group_idx, group in enumerate(['Control', 'LOPE', 'EOPE']):
            subset = dataframe[(dataframe['Timepoint'] == timepoint) & (dataframe['Group'] == group)]
            if not subset.empty:
                # Calculate interquartile range (IQR) and determine outlier thresholds
                q1 = subset[f'Average Mean {feature}'].quantile(0.25)
                q3 = subset[f'Average Mean {feature}'].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                # Find outliers based on thresholds
                outliers = subset[
                    (subset[f'Average Mean {feature}'] < lower_bound) | 
                    (subset[f'Average Mean {feature}'] > upper_bound)
                ]

                # Annotate each outlier near its respective data point
                for _, row in outliers.iterrows():
                    # Calculate adjusted x_position for the current group within the timepoint
                    x_position = timepoint_idx + (group_idx - 1) * 0.25  # -1, 0, +1 for Control, LOPE, EOPE

                    plt.text(
                        x=x_position + 0.08,  # Adjust horizontal position slightly
                        y=row[f'Average Mean {feature}'] + 0.4/upper_bound,  # Adjust vertical position slightly
                        s=row['Sample Base'].split("-")[0] + "-" + row['Sample Base'].split("-")[1], 
                        fontsize=7,
                        color='black',
                        verticalalignment='center',
                        horizontalalignment='center',
                        rotation=30  # Add slight rotation for better readability
                    )

    # Add plot titles and labels
    plt.title(f"Average Mean {feature} by Timepoint and Group", fontsize=14)
    plt.xlabel("Timepoint", fontsize=12)
    plt.ylabel(f"Average Mean {feature}", fontsize=12)

    # Add a grid for improved readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout for better visualization
    plt.tight_layout()

    # Show the plot
    plt.show()

