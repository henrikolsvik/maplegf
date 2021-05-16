rule all:
    input:
        "results/results_lasso_interproscan.txt",
        "results/results_svm_interproscan.txt",
        "results/results_mlpnn_interproscan.txt",
        "results/results_random_forest_interproscan.txt",
        "results/results_xgboost_interproscan.txt",
        "results/results_lstm_interproscan.txt",
        "results/results_lasso_humann.txt",
        "results/results_svm_humann.txt",
        "results/results_mlpnn_humann.txt",
        "results/results_random_forest_humann.txt",
        "results/results_xgboost_humann.txt",
        "results/results_lstm_humann.txt",
        "preprocessing_output/processed_samples_interproscan.csv",
        "preprocessing_output/processed_samples_interproscan_coverage.csv",
        "preprocessing_output/unprocessed_samples_interproscan_term_count.csv",
        "preprocessing_output/processed_samples_interproscan_term_count.csv",
        "preprocessing_output/processed_samples_interproscan_parameter_data.csv",
        "preprocessing_output/processed_samples_humann.csv",
        "preprocessing_output/processed_samples_humann_coverage.csv",
        "preprocessing_output/unprocessed_samples_humann_term_count.csv",
        "preprocessing_output/processed_samples_humann_term_count.csv",
        "preprocessing_output/processed_samples_humann_parameter_data.csv",
        "preprocessing_output/metadata_interproscan.csv",
        "preprocessing_output/metadata_humann.csv"


rule preprocess_samples_interproscan:
    input:
        samp="samples/simplified/",
        meta="metadata/metadata.csv",
        config="configs/preprocessing_interproscan.txt"
    output:
        samp_out="preprocessing_output/processed_samples_interproscan.csv",
        stat_out="preprocessing_output/processed_samples_interproscan_coverage.csv",
        meta_out="preprocessing_output/metadata_interproscan.csv",
        term_count_unprocessed="preprocessing_output/unprocessed_samples_interproscan_term_count.csv",
        term_count_processed="preprocessing_output/processed_samples_interproscan_term_count.csv",
        parameter_data="preprocessing_output/processed_samples_interproscan_parameter_data.csv"
    priority: 2
    shell:
        "python scripts/term_processing.py {input.samp} {input.meta} {output.samp_out} {output.stat_out} {output.term_count_unprocessed} {output.term_count_processed} {output.parameter_data} {output.meta_out} {input.config}"

rule preprocess_samples_humann:
    input:
        samp="samples/go.tsv",
        meta="metadata/metadata_humann.csv",
        config="configs/preprocessing_humann.txt"
    output:
        samp_out="preprocessing_output/processed_samples_humann.csv",
        stat_out="preprocessing_output/processed_samples_humann_coverage.csv",
        meta_out="preprocessing_output/metadata_humann.csv",
        term_count_unprocessed="preprocessing_output/unprocessed_samples_humann_term_count.csv",
        term_count_processed="preprocessing_output/processed_samples_humann_term_count.csv",
        parameter_data="preprocessing_output/processed_samples_humann_parameter_data.csv"
    priority: 1
    shell:
        "python scripts/term_processing_humann.py {input.samp} {input.meta} {output.samp_out} {output.stat_out} {output.term_count_unprocessed} {output.term_count_processed} {output.parameter_data} {output.meta_out} {input.config}"


samples = "preprocessing_output/processed_samples_interproscan.csv"
samples_parameters = "preprocessing_output/processed_samples_interproscan_parameter_data.csv"
targets_ipr = "preprocessing_output/metadata_interproscan.csv"

targets_humann = "preprocessing_output/metadata_humann.csv"
samples_humann = "preprocessing_output/processed_samples_humann.csv"
samples_parameters_humann = "preprocessing_output/processed_samples_humann_parameter_data.csv"

rule ml_lasso:
    input:
        config="configs/lasso_simple.txt"
    output:
        results_txt = "results/results_lasso_interproscan.txt"
    shell:
        "python scripts/ml/lasso.py {samples} {samples_parameters} {targets_ipr} {output.results_txt} {input.config}"

rule ml_svm:
    input:
        config="configs/svm_simple.txt"
    output:
        results_txt = "results/results_svm_interproscan.txt"
    shell:
        "python scripts/ml/svm.py {samples} {samples_parameters} {targets_ipr} {output.results_txt} {input.config}"

rule ml_random_forest:
    input:
        config="configs/random_forest_simple.txt"
    output:
        results_txt = "results/results_random_forest_interproscan.txt"
    shell:
        "python scripts/ml/random_forest.py {samples} {samples_parameters} {targets_ipr} {output.results_txt} {input.config}"

rule ml_mplnn:
    input:
        config="configs/mlpnn_simple.txt"
    output:
        results_txt = "results/results_mlpnn_interproscan.txt"
    shell:
        "python scripts/ml/mlpnn.py {samples} {samples_parameters} {targets_ipr} {output.results_txt} {input.config}"

rule ml_xgboost:
    input:
        config="configs/xgboost_simple.txt"
    output:
        results_txt = "results/results_xgboost_interproscan.txt"
    shell:
        "python scripts/ml/xgboost_impl.py {samples} {samples_parameters} {targets_ipr} {output.results_txt} {input.config}"

rule ml_LSTM:
    input:
        config="configs/lstm_simple.txt"
    output:
        results_txt = "results/results_lstm_interproscan.txt"
    shell:
        "python scripts/ml/lstm.py {samples} {samples_parameters} {targets_ipr} {output.results_txt} {input.config}"

rule ml_lasso_humann:
    input:
        config="configs/lasso_simple.txt"
    output:
        results_txt = "results/results_lasso_humann.txt"
    shell:
        "python scripts/ml/lasso.py {samples_humann} {samples_parameters_humann} {targets_humann} {output.results_txt} {input.config}"

rule ml_svm_humann:
    input:
        config="configs/svm_simple.txt"
    output:
        results_txt = "results/results_svm_humann.txt",
    shell:
        "python scripts/ml/svm.py {samples_humann} {samples_parameters_humann} {targets_humann} {output.results_txt} {input.config}"

rule ml_random_forest_humann:
    input:
        config="configs/random_forest_simple.txt"
    output:
        results_txt = "results/results_random_forest_humann.txt",
    shell:
        "python scripts/ml/random_forest.py {samples_humann} {samples_parameters_humann} {targets_humann} {output.results_txt} {input.config}"

rule ml_mplnn_humann:
    input:
        config="configs/mlpnn_simple.txt"
    output:
        results_txt = "results/results_mlpnn_humann.txt",
    shell:
        "python scripts/ml/mlpnn.py {samples_humann} {samples_parameters_humann} {targets_humann} {output.results_txt} {input.config}"

rule ml_xgboost_humann:
    input:
        config="configs/xgboost_simple.txt"
    output:
        results_txt = "results/results_xgboost_humann.txt",
    shell:
        "python scripts/ml/xgboost_impl.py {samples_humann} {samples_parameters_humann} {targets_humann} {output.results_txt} {input.config}"

rule ml_LSTM_humann:
    input:
        config="configs/lstm_simple.txt"
    output:
        results_txt = "results/results_lstm_humann.txt"
    shell:
        "python scripts/ml/lstm.py {samples_humann} {samples_parameters_humann} {targets_humann} {output.results_txt} {input.config}"
