# Word-Level

This folder contains experiments for **word-level tokenization** of the informal-to-formal Persian text transformation task.  
Each notebook trains and evaluates two sequence-to-sequence models:

- **LSTM-based Seq2Seq**
- **Transformer-based Seq2Seq**

Both models treat the problem as a **machine translation task**, mapping informal Persian sentences to their formal equivalents.


## Contents
- `lstm_word.ipynb` → LSTM-based seq2seq training & evaluation
- `transformer_word.ipynb` → Transformer-based seq2seq training & evaluation

Output files (logs, results, checkpoints) are uploaded to Google Drive:  
[Drive Link](https://drive.google.com/drive/folders/1I6kPQQLo8HPzagEZWFO2mQqmF1SLa4S9?usp=sharing)


## Results

| Model         | Variant   | Train CER | Train WER | Val CER | Val WER | Test CER | Test WER |
|---------------|-----------|-----------|-----------|---------|---------|----------|----------|
| **LSTM**      | Best Loss | 90.72%    | 120.34% = 100%   | 100.57% = 100%  | 131.91% = 100% | 100.49% = 100%  | 131.67% = 100%   |
| **LSTM**      | Best CER  | 73.58%    | 87.63%    | 72.92%  | 88.16%  | 72.85%   | 87.91%   |
| **Transformer** | Best Loss | 12.76    | 15.29    | 51.50%  | 63.39%  | 50.47%   | 62.39%   |
| **Transformer** | **Best CER**  | **10.40%**    | **12.34%**    | **46.98%**  | **58.93%**  | **46.59%**   | **58.10%**   |


## Notes
- Resulted evaluation CSVs are available in the Drive folder for analysis.
- See the [main README](../README.md) for the overall project description and dataset details.
