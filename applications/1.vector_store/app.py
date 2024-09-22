from vector_store import VectorStore
import numpy as np

# Create a vector store instance
vector_store = VectorStore()

# Define your sentences
sentences = [
    "I eat Mango",
    "mango is my favorite fruit",
    "mango, apple, and oranges are fruits",
    "fruits are good for health"
]

#  Tokenization and vocabulary creation

vocabulary = set()

for sentence in sentences:
    tokens = sentence.lower().split()
    vocabulary.update(tokens)

# Assign unique indices to words in the vocab
word_to_index = {word: i for i, word in enumerate(vocabulary)}

# Vectorization
sentence_vectors = {}

for sentence in sentences:
    vector = np.zeros(len(vocabulary))
    tokens = sentence.lower().split()
    for token in tokens:
        vector[word_to_index[token]] += 1

    sentence_vectors[sentence] = vector

# Add the vectors to the vector store
for sentence, vector in sentence_vectors.items():
    vector_store.add_vector(sentence, vector)

# searching for the similarity
query_sentence = "Mango is the best fruit"
query_vector = np.zeros(len(vocabulary))
query_tokens = query_sentence.lower().split()
for token in query_tokens:
    if token in word_to_index:
        query_vector[word_to_index[token]] += 1

similar_sentences = vector_store.find_similar_vectors(
    query_vector, num_results=2)
print("Query Sentence:", query_sentence)
print("Similar Sentences:")
for sentence, similarity in similar_sentences:
    print(f"{sentence} : Similarity = {similarity: .4f}")
