
# Semantic Checker API

The `semantic_checker_api.py` file sets up an API using FastAPI to compute semantic similarity between pairs of sentences. It utilizes a pre-trained Universal Sentence Encoder model from TensorFlow Hub, which is loaded from a specified path. The API includes the following key components:

- **Imports**: The script imports necessary libraries such as TensorFlow, Pandas, NumPy, FastAPI, and Pydantic.

- **Model Loading**: The Universal Sentence Encoder model is loaded using TensorFlow's `tf.saved_model.load()` function.

- **CORS Middleware**: Cross-Origin Resource Sharing (CORS) middleware is configured to allow requests from specified origins, facilitating interaction with the API from different domains.

- **SentencePair Class**: A Pydantic model `SentencePair` is defined to handle input data, which consists of two sentences and Paragraphs whose similarity score needs to be computed to check if they sound same in thier meanings or not even after complete rephrasing.

- **Embedding Function**: The `embed()` function takes input sentences and returns their embeddings using the loaded model.

- **Similarity Endpoint**: A POST endpoint `/similarity` is defined, which accepts a `SentencePair` object, computes the cosine similarity between the embeddings of the two sentences, and returns the similarity score.

The file essentially provides a simple yet powerful API that leverages machine learning to assess the semantic similarity of sentence pairs, useful for applications like duplicate detection, paraphrase identification, and more.

