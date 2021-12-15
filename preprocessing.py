from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

STOPWORDS = set(stopwords.words('danish'))
MIN_WORDS = 2
MAX_WORDS = 200

PATTERN_S = re.compile("\'s")  # matches `'s` from text
PATTERN_RN = re.compile("\\r\\n")  # matches `\r` and `\n`
PATTERN_PUNC = re.compile(r"[^\w\s]")  # matches all non 0-9 A-z whitespace


def clean_text(text):
    #Series of cleaning. String to lower case, remove non words characters and numbers (punctuation, curly brackets etc).
    text = text.lower()  # lowercase text
    # replace the matched string with ' '
    text = re.sub(PATTERN_S, ' ', text)
    text = re.sub(PATTERN_RN, ' ', text)
    text = re.sub(PATTERN_PUNC, ' ', text)
    return text


def tokenizer(sentence, min_words=MIN_WORDS, max_words=MAX_WORDS, stopwords=STOPWORDS, lemmatize=True):
    #Lemmatize, tokenize, crop and remove stop words.
    if lemmatize:
        stemmer = WordNetLemmatizer()
        tokens = [stemmer.lemmatize(w) for w in word_tokenize(sentence)]
    else:
        tokens = [w for w in word_tokenize(sentence)]
    token = [w for w in tokens if (len(w) > min_words and len(w) < max_words
                                   and w not in stopwords)]
    return tokens


def clean_sentences(df):
    #Remove irrelavant characters (in new column clean_sentence).
    print('Cleaning sentences...')
    df['clean_sentence'] = df['sentence'].apply(clean_text)
    df['tok_lem_sentence'] = df['clean_sentence'].apply(
        lambda x: tokenizer(x, min_words=MIN_WORDS, max_words=MAX_WORDS, stopwords=STOPWORDS))
    return df

