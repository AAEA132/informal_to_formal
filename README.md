# Converting Informal Persian Text to Formal Persian  

## Overview  
This project explores the task of **converting informal Persian text into formal Persian** using deep learning models.  I approached this task as a **sequence-to-sequence (seq2seq) problem**, similar to **machine translation** — where the "source language" is informal Persian and the "target language" is formal Persian.

With this approach in mind, I implemented and compared **two widely used seq2seq architectures**:
- **Seq2Seq LSTMs** (a classical recurrent approach)  
- **Transformers** (state of the art attention-based models)  

Both were tested with three tokenization approaches: **character-level**, **word-level**, and **subword-level**.  

Combining these two models with the mentioned tokenization approaches created a six-part comparative study:  

- **Char + LSTM**  
- **Char + Transformer**  
- **Word + LSTM**  
- **Word + Transformer**  
- **Subword + LSTM**  
- **Subword + Transformer**  


## Motivation  
A large portion of Persian text available online is written informally, making it difficult to use directly for NLP tasks. By transforming informal Persian into its formal form, this project provides a **preprocessing step** which can be used for building better Persian NLP tools and training large language models on cleaner and more standardized data.


## Dataset  
I used the dataset presented in **Developing an Informal-Formal Persian Corpus**. You can find full details in the paper: https://aclanthology.org/2025.abjadnlp-1.6/  

- **Size:** ~50,000 informal-formal sentence pairs  
- **Sources:** Social media, messaging apps, websites, blogs, books, movies  
- **Characteristics:**  
  - Lexical and syntactic variation (50% require structural changes)  
  - Wide coverage of colloquial Persian  
  - Variable sentence lengths   

**Example:**  

| Informal | Formal |  
|----------|--------|  
| من دوس دارم برم خونه درس بخونم. | من دوست دارم که به خانه بروم تا درس بخوانم. |  
| میتونی منو ببری خونمون یکم نون وردارم؟ | می‌توانی من را به خانه‌مان ببری تا کمی نان بردارم؟ |


## Models & Tokenization  

### 🔹 Why LSTMs and Transformers?  
Since the task is framed as **sequence-to-sequence transformation**, the natural model choices were:  

- **LSTM-based Seq2Seq models** → classical architecture for machine translation, capable of handling variable-length input and output sequences.  
- **Transformer models** → more recent architecture relying on attention mechanisms, better at capturing long-range dependencies and parallelizing training.  

This allowed me to compare a **baseline seq2seq approach (LSTMs)** against the **modern state-of-the-art (Transformers)** in the context of Persian text transformation.  

### 🔹 Seq2Seq LSTM  
- Embeddings, multi-layer LSTM encoder/decoder, dropout, fully connected output.  

### 🔹 Transformer  
- Implemented using `nn.Transformer` (PyTorch).  
- Positional encodings, padding & look-ahead masks.  

### 🔹 Tokenization Methods  
- **Character-level** → small vocab, long sequences, weak semantics.  
- **Word-level** → strong semantics, large vocab, OOV problems.  
- **Subword-level** (SentencePiece Unigram) → balance of vocabulary size + semantic coverage.  


## Results  

### Metrics  
- **CER (Character Error Rate):** measures the percentage of characters that are wrong compared to the reference.  
- **WER (Word Error Rate):** measures the percentage of words that are wrong compared to the reference.  
Lower is better for both.  

### Summary (Test Set Averages)  

| Tokenization | Model | CER (%) | WER (%) |  
|--------------|-------|-----|-----|  
| Char | LSTM | 98.75% | 126.03% = 100% |  
| **Char** | **Transformer** | **15.44%** | **26.11%** |  
| Subword | LSTM | 59.41% | 79.64% |  
| **Subword** | **Transformer** | **32.58%** | **43.24%** |  
| Word | LSTM | 72.85% | 87.91% |  
| Word | Transformer | 46.59% | 58.10% |  

> These are the **best test-set results** obtained among all model checkpoints (for each tokenization + model type).  
> In each tokenization folder/notebook, you can find additional models saved by best validation loss and best validation CER, covering evaluation results for training, validation, and test sets.  

✅ **Best performing models:**  
- **Char-level Transformer** (lowest CER/WER overall)  
- **Subword-level Transformer** (best balance between vocabulary size & semantic handling)  


## Example Outputs  

| Input (Informal) | Target (Formal) | Prediction (Transformer-Subword best test-set results) |  CER | WER |
|------------------|-----------------|---------------------------------|-----|-----| 
| عمدا میخواد حرصمون بده | عمدا می‌خواهد ما را حرص بدهد | عمدا می خواهد ما را حرص بدهد | 0% | 0% |
کلاهتو سف بچسب همسایتو دزد نکن | کلاهت را سفت بچسب و همسایه‌ات را دزد نکن | کلاهت را زنبور عسل همسایه ات را دزد نکن | 25% | 30% |

## Reflections  

This project gave me hands-on experience in:  

- Building **NLP pipelines from scratch** (data preprocessing, tokenization, batching, padding).  
- Implementing **seq2seq models** with LSTMs and Transformers in PyTorch.  
- Handling **evaluation metrics (CER, WER)** and saving/loading best checkpoints.  
- Debugging model behaviors (e.g., handling `<BOS>`/`<EOS>` tokens, masking).  

**Key insights:**  
- Transformers were significantly better than LSTMs for this task.  
- Tokenization strongly affects results — **subword tokenization gives the best balance**.  
- Character-level Transformers surprisingly achieved the best raw performance, likely due to handling spelling variants in informal Persian.  

**Future work:**  
- Implement **beam search decoding** instead of greedy decoding.  
- Use **pretrained embeddings** or pretrained language models (e.g., multilingual BERT).  

## Running the Code  

    All parts are Jupyter Notebooks (Ran on Google Colab).  
    Place the dataset Excel file either:  
      - directly in Colab via upload, or  
      - in the same directory as the notebook (If run locally).  

**Dependencies:**  
- Python  
- PyTorch  
- Jiwer  
- SentencePiece  
- NumPy, Pandas  


## Repository Structure  

    char/         → lstm_char.ipynb, transformer_char.ipynb
    subword/      → lstm_subword.ipynb, transformer_subword.ipynb
    word/         → lstm_word.ipynb, transformer_word.ipynb
    exportStatements.xlsx → dataset 