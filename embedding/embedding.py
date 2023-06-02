import numpy as np

sentence = 'The quick brown fox jumps over the lazy dog'

tokens = sentence.split()
vocabulary = sorted(set(tokens))

word_to_index = {word: i for i, word in enumerate(vocabulary)}
index_to_word = {i: word for i, word in enumerate(vocabulary)}

def generate_embeddings(vocabulary_size, embedding_size):
    return np.random.randn(vocabulary_size, embedding_size)

vocabulary_embeddings = generate_embeddings(len(vocabulary), 4)

word_to_embedding = {word: vocabulary_embeddings[word_to_index[word]] for word in vocabulary}

print(word_to_embedding['quick']) # [0.90063452 -0.17414497  0.34447976 -0.05756612]

# Key insight
# We can represent a token, such as a word as a vector.
# The initial values for the vector are random.
# The goal is then to find the values for the vector which best represents the token.
# This can be understood as a way of capturing the meaning of the token.
# In practice the meaning can be learned through the training of a model.
# That is to say that we can include the vector as a parameter in a model, and adjust it with backpropagation.


# In a well trained embedding space optimized for word similarity, we would expect the following to be true:
# word_to_embedding['king'] - word_to_embedding['man'] + word_to_embedding['woman'] = word_to_embedding['queen']