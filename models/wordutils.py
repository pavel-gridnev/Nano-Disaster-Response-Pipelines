from nltk import word_tokenize, WordNetLemmatizer


def tokenize(text):
    """
    Tokenize free text by lower casing text, splitting into words, lemmatazing separate words
    :param text: free text
    :return: list of tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens
