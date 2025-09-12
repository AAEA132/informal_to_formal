# Subword-Level

This folder contains experiments for **subword-level tokenization** of the informal-to-formal Persian text transformation task.  
Each notebook trains and evaluates two sequence-to-sequence models:

- **LSTM-based Seq2Seq**
- **Transformer-based Seq2Seq**

Both models treat the problem as a **machine translation task**, mapping informal Persian sentences to their formal equivalents.


## Contents
- `lstm_subword.ipynb` → LSTM-based seq2seq training & evaluation
- `transformer_subword.ipynb` → Transformer-based seq2seq training & evaluation

Output files (logs, results, checkpoints) are uploaded to Google Drive:  
[Drive Link](https://drive.google.com/drive/folders/1U9Z1W4rVk8X9e0ZOhWtTE1OugQ_Pht_0?usp=sharing)


## Results

| Model         | Variant   | Train CER | Train WER | Val CER | Val WER | Test CER | Test WER |
|---------------|-----------|-----------|-----------|---------|---------|----------|----------|
| **LSTM**      | Best Loss | 45.39%    | 60.47%    | 59.79%  | 79.95%  | 59.41%   | 79.64%   |
| **LSTM**      | Best CER  | 44.45%    | 59.30%    | 59.75%  | 80.10%  | 59.70%   | 80.17%   |
| **Transformer** | **Best Loss** | **8.74%**    | **10.83%**    | **32.49%**  | **42.80%**  | **32.58%**   | **43.24%**   |
| **Transformer** | Best CER  | 10.05%    | 12.57%    | 33.15%  | 43.81%  | 33.04%   | 43.73%   |


## Notes
- Resulted evaluation CSVs are available in the Drive folder for analysis.
- See the [main README](../README.md) for the overall project description and dataset details.
