import pandas as pd
from src import ROOT_DIR
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Optional: Download NLTK data (if not already available)
import nltk


def preprocess_text(text):
    """
    Preprocesses the input text by performing several operations:
    - Lowercasing: Convert the text to lowercase to maintain consistency.
    - Tokenization: Split the text into individual words.
    - Removing stopwords: Optionally remove common words that may not be necessary for understanding the meaning.
    - Removing punctuation: Strip punctuation as it may not be needed.
    - Removing special characters: Remove any unwanted special characters.
    - Whitelist characters: Optionally, keep only alphanumeric characters.

    Args:
    - text (str): The text to be preprocessed.

    Returns:
    - str: The preprocessed text.
    """
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers and special characters, only keep letters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords (optional)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Re-join tokens back to a single string
    text = ' '.join(tokens)

    return text


def main(path: str = f'{ROOT_DIR}/data/filtered.tsv'):
    # Step: 1. reading
    df = pd.read_csv(path, sep='\t', header=0)

    # Step: 2. removing redundant columns
    df = df.drop(columns=['similarity', 'lenght_diff', 'Unnamed: 0'])

    # Step: 3. swapping to get real data
    mask = df['ref_tox'] < df['trn_tox']

    # Swap the values where the condition is True
    df.loc[mask, ['reference', 'translation']] = df.loc[mask, ['translation', 'reference']].values

    # Step: 4. Preprocessing
    df['reference'] = df['reference'].apply(preprocess_text)
    df['translation'] = df['translation'].apply(preprocess_text)

    # Step: 5. Saving
    df.to_csv(f'{ROOT_DIR}/data/filtered_and_prepocessed.csv', index=False)


if __name__ == '__main__':
    # !!! Run it if you see some problems with nltk !!!
    # import ssl
    #
    # try:
    #     _create_unverified_https_context = ssl._create_unverified_context
    # except AttributeError:
    #     pass
    # else:
    #     ssl._create_default_https_context = _create_unverified_https_context
    # nltk.download('punkt')
    # nltk.download('stopwords')
    main()
