## MAPLEgf - Multi-Algorithm Pipeline for machine LEarning on Gene Functions
This is the github repo for MAPLEgf a multi-algorithm pipeline for machine learning on gene functions.  
MAPLEgf currently supports IPR and GO terms from InterProScan and GO terms from HUMAnN.

### Installation Instructions

1. Clone this repo.
2. Install a supported of Python (3.6 - 3.8, confirmed to work as of 14th of May 2021).
3. Install all dependencies by running `pip install -r requirements.txt`.
4. (optional) Make directories for samples and metadata `mkdir samples && mkdir metadata`.

### Running the program

Take the results from functional gene annotation from InterProScan and HUMAnN and place them in `samples/simplified/` and `samples/go.tsv` respectively. MAPLEgf expects both to be present. MAPLEgf also expects a metadata file in the .csv format which can match with the sample name strings. The format expected is no headers, column 1 for sample names, column 2 for labels. One sample per row. If running only one of the two supported algorithms, please edit the `Snakemake.smk` file by commenting out all rules relating to the one you do not have present. In this file you can also edit the expected input filenames and locations in the preprocessing rules. 

Run the program by running `snakemake`.
If a TSD slurm node you can run `snakemakejob.sh`

As MAPLEgf makes use of snakemake it "listens" for changes in the file tree to trigger reruns. As such in order to perform an analysis on a certain algorithm again, the config file for the algorithms needs to be changed or the `results/results_<algorithm_name>_<dataset>.txt` file must be removed or renamed.

### Configuring MAPLEgf

The program comes with a set of configurations under `configs/`.
Here you can set the individual parameters for each algorithm and for the preprocessing stages.
In each config file there are a set of short instructions for what each algorithm does as well as a list of default parameters.
Settings for normalization, univariate feature selection and interpretation are set in each algorithm config file.

### Results

The results of the executions are written to files in the results folder and aggregated in the `results/combined_results.csv` file.
If interpretation is enabled it will yield a series of files for each k-fold that is interpreted. For k >= 2 there will also be generated a combined file for the averaged results of the interpretations.   

### Dataflow diagram

An overview of the dataflow in MAPLEgf.

![figure3_maplegf](https://user-images.githubusercontent.com/17406317/118392550-80313900-b63a-11eb-8424-dbf33f85f911.png)







