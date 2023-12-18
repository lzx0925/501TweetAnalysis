# -*- coding: utf-8 -*-

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.cluster import SpectralClustering

file_path1 = 'IRAhandle_tweets_1.csv'
file_path2 = 'IRAhandle_tweets_2.csv'
file_path3 = 'IRAhandle_tweets_3.csv'
file_path4 = 'IRAhandle_tweets_4.csv'

sample_data1 = pd.read_csv(file_path1)
sample_data2 = pd.read_csv(file_path2)
sample_data3 = pd.read_csv(file_path3)
sample_data4 = pd.read_csv(file_path4)

sample_data = pd.concat([sample_data1,sample_data2,sample_data3,sample_data4])
sample_data.head()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)
cleaned_data = None
try:
  print("try here")
  cleaned_file = pd.read_csv('cleaned_data.csv')
except:
  cleaned_data = sample_data[['content']]
  cleaned_data['content'] = cleaned_data['content'].str.lower()
  cleaned_data['content'] = cleaned_data['content'].str.replace(r'http\S+', '', regex=True)
  cleaned_data['content'] = cleaned_data['content'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
  print("222222222")
  nltk.download('punkt')
  nltk.download('stopwords')
  nltk.download('wordnet')

  lemmatizer = WordNetLemmatizer()

  cleaned_data = cleaned_data.dropna(subset=['content'])
  cleaned_data['content'] = cleaned_data['content'].apply(preprocess_text)

  cleaned_data.to_csv('cleaned_data.csv', index=False, encoding='utf-8')

  cleaned_data.head()
else:
  print("file exists")
  cleaned_data = cleaned_file
  cleaned_data = cleaned_data.dropna(subset=['content'])


# from sklearn.feature_extraction.text import TfidfVectorizer

# # Create a TF-IDF Vectorizer instance
# tfidf_vectorizer = TfidfVectorizer(max_features=2500)  # Limiting to 500 features for demonstration

# # Apply TF-IDF to the 'content' column
# tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_data['content'])

# # Convert to a dense matrix
# tfidf_dense_matrix = tfidf_matrix.todense()

# # Retrieve and display feature names (words)
# feature_names = tfidf_vectorizer.get_feature_names_out()
# for name in feature_names:
#   print( name)

# feature_names = pd.DataFrame(feature_names, columns=['Feature Name'])
# feature_names.to_csv('feature_names.csv', index=False, encoding='utf-8')

# # Displaying the shape of the matrix
# print("Shape of TF-IDF Matrix:", tfidf_dense_matrix.shape)

# # Display a portion of the dense matrix (first 5 rows and 10 columns for example)
# # This provides a glimpse into which features are present in these documents
# print("Sample of TF-IDF Matrix:\n", tfidf_dense_matrix[:5, :10])

# """# Embedding Training"""

# from nltk.tokenize import word_tokenize
# from gensim.models import Word2Vec
# import pandas as pd

# # Load the provided CSV file to examine the feature names
# feature_names_df = None

# if feature_names is not None:
#   file_path = 'feature_names.csv'
#   feature_names_df = pd.read_csv(file_path)
# else:
#   feature_names_df = feature_names

print("start tokenizing...")
tokenized_content = [word_tokenize(doc) for doc in cleaned_data['content']]

from gensim.models import Word2Vec

word2vec_model = Word2Vec(sentences=tokenized_content, vector_size=100, window=5, min_count=2, workers=4)

word2vec_model.save("word2vec_tweet_model.model")

print("complete model training")
"""# Visualization"""

positive_words = [
    "awesome", "amazing", "perfect", "beautiful", "pretty", "adorable", "proud", "glad", "best",
    "sweet", "clear", "brilliant", "absolutely", "famous", "wild", "fresh", "dark", "light", "smart",
    "enjoy", "winning", "brave", "popular", "innocent", "grand", "better", "greatest"
]

negative_words = [
    "fuck", "bitch", "dick", "crazy", "insane", "dirty", "crap", "dumb", "pathetic", "disgusting",
    "rediculous", "awful", "horrible", "wrong", "bitter", "angry", "badly", "broken", "collapse",
    "poor", "corrupt", "critical", "guilty", "brutality", "chaos", "fool", "fails", "destroy", "fake",
    "false", "killed", "accused", "allegation", "arrest", "brutal", "appeal", "exclusive", "illegal",
    "criminal", "fraud", "crooked", "boycott", "backlash", "bullshit", "assault","apologize", "bad",
    "bias", "condemn", "disaster","disgraceful", "evil", "exposed","enemy", "dead", "fat"
]
print(positive_words)
print(negative_words)

bias_dict ={
  "political_words":
   [
    "afd", "altleft", "antitrump", "armed", "assassination", "authority", "ballot", "barackobama", "barack", "cia",
    "communist", "democracy",
    "democrat", "democratic", "diversity", "donald", "election",
    "gov", "government", "governor",
    "antitrump", "debalwaystrump",
    "protrump", "realdonaldtrump", "trump", "trumppence",
    "trumpsfavoriteheadline", "trumptrain","republican","liberal","liberty","altleft", "left", "leftist",
    "right","white","whitehouse","feminist"
],
  "gender_words" :[
    "abortion", "dad", "daughter", "drug", "father", "female", "gay",
    "gender", "girl", "girlfriend", 'actress', 'boyfriend', 'bride', 'brother',
    'husband', 'lady', 'male', 'man', 'mom', 'mother', 'sister', 'son', 'wife', 'woman'
  ],
  "religious_words" : [
    "christian", "church", "gay",
    'bible', 'bishop', 'cleric', 'faith', 'imam', 'minister', 'mosque', 'pastor', 'prayer',
    'priest', 'rabbi', 'religion', 'religious', 'sermon', 'synagogue', 'temple',
    "islam", "islamic", "islamkills", "stopislam","muslim", "kkk", "terror", "terrorism", "terrorist",
  ],
  "racial_words" : [
    "african", "amis", "blackhistorymonth", "blacklivesmatter", "blackmatters", "blackskinisnotacrime", "blacktolive",
    "blacktwitter", "blm", "chinese", "citizen", 'race', 'ethnic', 'african', 'asian', 'caucasian', 'latino', 'black', 'white']
}



from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def get_trained_plot(words, positive_words, negative_words, political_words, title):
    word_vectors = np.array([word2vec_model.wv[word] for word in words])
    n_clusters = 10
    spectral_cluster = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=0)
    clusters = spectral_cluster.fit_predict(word_vectors)

    tsne = TSNE(n_components=2, random_state=0, perplexity=40)
    word_vectors_tsne = tsne.fit_transform(word_vectors)


    plt.figure(figsize=(10, 6))
    for i, word in enumerate(words):
        plt.scatter(word_vectors_tsne[i, 0], word_vectors_tsne[i, 1], c=f'C{clusters[i]}')
        plt.annotate(word, (word_vectors_tsne[i, 0], word_vectors_tsne[i, 1]))

    plt.title(title)
    plt.show()

get_trained_plot(set(bias_dict["political_words"] + positive_words + negative_words), positive_words, negative_words, bias_dict["political_words"], "self-trained political bias")

get_trained_plot(set(bias_dict["gender_words"] + positive_words + negative_words), positive_words, negative_words, bias_dict["gender_words"], "self-trained gender bias")

get_trained_plot(set(bias_dict["religious_words"] + positive_words + negative_words), positive_words, negative_words, bias_dict["religious_words"], "self-trained religious bias")

get_trained_plot(set(bias_dict["racial_words"] + positive_words + negative_words), positive_words, negative_words, bias_dict["racial_words"], "self-trained racial bias")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_glove_model(file_path):
    glove_model = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            try:
                embedding = np.array([float(val) for val in split_line[1:]])
                glove_model[word] = embedding
            except ValueError:
                continue
    return glove_model

def plot_embeddings(embeddings, words, positive_words, negative_words, political_words, title):
    plt.figure(figsize=(10, 6))
    for i, word in enumerate(words):
        plt.scatter(embeddings[i, 0], embeddings[i, 1], c=f'C{clusters[i]}')
        plt.annotate(word, (embeddings[i, 0], embeddings[i, 1]))

    plt.title(title)
    plt.show()

glove_file_path = 'glove.twitter.27B.100d.txt'

glove_model = load_glove_model(glove_file_path)

political_filter = [word for word in bias_dict["political_words"] if word in glove_model]
overall_filter = [word for word in (bias_dict["political_words"] + positive_words + negative_words) if word in glove_model]

glove_embeddings = np.array([glove_model[word] for word in overall_filter if word in glove_model])
n_clusters = 10
spectral_cluster = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=0)
clusters = spectral_cluster.fit_predict(glove_embeddings)

tsne = TSNE(n_components=2, random_state=0, perplexity=5)
glove_embeddings_tsne = tsne.fit_transform(glove_embeddings)

plot_embeddings(glove_embeddings_tsne, bias_dict["gender_words"] + positive_words + negative_words, positive_words, negative_words, bias_dict["gender_words"], "GloVe Embeddings: Gender Bias")
plot_embeddings(glove_embeddings_tsne, bias_dict["religious_words"] + positive_words + negative_words, positive_words, negative_words, bias_dict["religious_words"], "GloVe Embeddings: Religious Bias")
plot_embeddings(glove_embeddings_tsne, bias_dict["racial_words"] + positive_words + negative_words, positive_words, negative_words, bias_dict["racial_words"], "GloVe Embeddings: Racial Bias")
plot_embeddings(glove_embeddings_tsne, overall_filter, positive_words, negative_words, political_filter, "GloVe Embeddings: Political Bias")