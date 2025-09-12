# Character-Level

This folder contains experiments for **character-level tokenization** of the informal-to-formal Persian text transformation task.  
Each notebook trains and evaluates two sequence-to-sequence models:

- **LSTM-based Seq2Seq**
- **Transformer-based Seq2Seq**

Both models treat the problem as a **machine translation task**, mapping informal Persian sentences to their formal equivalents.


## Contents
- `lstm_char.ipynb` → LSTM-based seq2seq training & evaluation
- `transformer_char.ipynb` → Transformer-based seq2seq training & evaluation

Output files (logs, results, checkpoints) are uploaded to Google Drive:  
[Drive Link](https://drive.google.com/drive/folders/1L3cfYZNZHHIx67DlljzoLNQEdfAyNGn1?usp=sharing)


## Results

| Model         | Variant   | Train CER | Train WER | Val CER | Val WER | Test CER | Test WER |
|---------------|-----------|-----------|-----------|---------|---------|----------|----------|
| **LSTM**      | Best Loss | 98.44%    | 125.03% = 100%    | 98.83  | 125.46% = 100%  | 98.75%   | 126.03% = 100%   |
| **LSTM**      | Best CER  | 96.07%    | 136.55% = 100%    | 197.09% = 100%  | 137.02% = 100%  | 193.13% = 100%   | 138.94% = 100%   |
| **Transformer** | **Best Loss** | **13.81%**    | **22.88%**    | **15.27%**  | **25.79%**  | **15.44%**   | **26.11%**   |
| **Transformer** | Best CER  | 19.03%    | 33.51%    | 19.35%  | 34.36%  | 19.66%   | 35.04%   |


## Notes
- Resulted evaluation CSVs are available in the Drive folder for analysis.
- See the [main README](../README.md) for the overall project description and dataset details.
