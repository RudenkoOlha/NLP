# FilmTV Movie Genre Predictor

Predicting a film's genre from its plot description, and comparing a simple baseline, a recurrent model and a fine-tuned BERT.

A university NLP course project by Olha Rudenko and Anna Trehub.

## Data
- **FilmTV Movies Dataset** (Kaggle, ~41,000 films): training and evaluation. After cleaning, 39,762 samples remain; the 10 most frequent genres give 34,965 samples (70% train / 15% validation / 15% test).
- **TMDB 5000 Movies** (Kaggle): independent test set for generalisation (4,717 samples after preprocessing).

The classes are imbalanced (Drama and Comedy dominate), which is taken into account when interpreting the results.

## Methods
- Text preprocessing: tokenisation, stop-word removal, lemmatisation (NLTK).
- **Model 1:** linear layer on averaged GloVe 50D embeddings (baseline).
- **Model 2:** bidirectional GRU (2 layers, hidden size 128) on GloVe sequences.
- **Model 3:** fine-tuned `bert-base-uncased` (Hugging Face Transformers), 5 epochs, lr = 2e-5.
- Metrics: accuracy and Top-3 accuracy.
- Extra analysis: semantic similarity graph of 400 films (cosine similarity), community detection and clustering coefficient.

## Results (FilmTV test set)
| Model | Accuracy | Top-3 accuracy |
|---|---|---|
| GloVe average (baseline) | 0.512 | 0.852 |
| GRU + GloVe | 0.520 | 0.852 |
| BERT fine-tuning | 0.619 | 0.909 |

On TMDB 5000 (no retraining) accuracy drops by about 0.08-0.09 for all models (BERT: 0.529).

BERT performs best thanks to contextual embeddings. The high Top-3 accuracy suggests that many films naturally fit several genres, so single-label classification forces an arbitrary choice.

## Run
Open `movie_genre_predictor.ipynb` in Google Colab. The notebook expects the two Kaggle datasets and GloVe 50D vectors to be uploaded. Python, PyTorch, Transformers, NLTK, scikit-learn, NetworkX.
