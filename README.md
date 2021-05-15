## MAPLEgf - Multi-Algorithm Pipeline for machine LEarning on Gene Functions
This is the github repo for MAPLEgf a multi-algorithm pipeline for machine learning on gene functions.
MAPLEgf currently supports IPR and GO terms from InterProScan and GO terms from HUMAnN.

### Installation Instructions

1. Clone this repo
2. Install a supported of Python (3.6 - 3.8, confirmed to work as of 14th of May 2021)
3. Install all dependencies by running `pip install -r requirements.txt`
4. (optional) Make directories for samples and metadata `mkdir samples && mkdir metadata`.

### Running the program

Take the results from functional gene annotation from InterProScan and HUMAnN and place them in `samples/simplified/` and `samples/go.tsv` respectively. MAPLEgf expects both to be present. If running only one of the two supported algorithms, please edit the `Snakemake.smk` file by commenting out all rules relating to the one you do not have present. In this file you can also edit the inputfile names and locations in the preprocessing rules.

Run the program by running `snakemake`

As MAPLEgf makes use of snakemake.

### Configuring MAPLEgf

The program comes with a set of configurations under.



