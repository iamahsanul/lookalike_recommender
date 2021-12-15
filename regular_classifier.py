from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from preprocessing import *

def extract_best_indices(m, topk, mask=None):
    #Use sum of the cosine distance over all tokens ans return best mathes.
    # return the sum on all tokens of cosinus for each sentence
    if len(m.shape) > 1:
        cos_sim = np.mean(m, axis=0)
    else:
        cos_sim = m
    index = np.argsort(cos_sim)[::-1] # from highest idx to smallest score
    if mask is not None:
        assert mask.shape == m.shape
        mask = mask[index]
    else:
        mask = np.ones(len(cos_sim))
    mask = np.logical_or(cos_sim[index] != 0, mask) #eliminate 0 cosine distance
    best_index = index[mask][:topk]
    return best_index


def get_recommendations_tfidf(vectorizer, sentence, tfidf_mat):
    # Embed the query sentence
    tokens_query = [str(tok) for tok in tokenizer(sentence)]
    embed_query = vectorizer.transform(tokens_query)
    # Create list with similarity between query and dataset
    mat = cosine_similarity(embed_query, tfidf_mat)
    # Best cosine distance for each token independantly
    best_index = extract_best_indices(mat, topk=1)
    return best_index


def test_regular_classifier(train_features, test_features, train_labels, test_labels):
    stop_words = set(stopwords.words('danish'))
    token_stop = tokenizer(' '.join(stop_words), lemmatize=False)

    # Fit TFIDF
    vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer)
    tfidf_mat = vectorizer.fit_transform(train_features['sentence'].values)


    test_probability = []
    predict_probability = []

    # Return the best match between query and dataset
    for sentence, prob in zip(test_features['sentence'], test_labels['seed']):
        test_probability.append(prob)
        predict_index = get_recommendations_tfidf(vectorizer, sentence, tfidf_mat)
        predict_probability.append(((train_labels[['seed']].iloc[predict_index]).values[0])[0])

    #test_sentence = 'an example data point'
    #best_index = get_recommendations_tfidf(vectorizer, test_sentence, tfidf_mat)
    return test_probability, predict_probability
    #print(df[['original_title', 'genres', 'sentence']].iloc[best_index])


