from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def custom_top10(queries, docs, model):

    # Convert the queries and documents to embeddings
    query_embeddings = model.encode(queries)
    doc_embeddings = model.encode(docs)

    # Initialize an empty list to store the top 10 documents for each query
    top10_docs = []

    # For each query, calculate the cosine similarity with each document
    for query in query_embeddings:
        similarities = cosine_similarity([query], doc_embeddings)

        # Get the top 10 document indices in descending order of similarity
        top10_indices = np.argsort(similarities[0])[::-1][:10]

        # Append the indices to the list
        top10_docs.append(list(top10_indices))

    return top10_docs


def check_information(contexts, query, model):

    # Convert the contexts and query to embeddings
    context_embeddings = model.encode(contexts)
    query_embedding = model.encode([query])[0]

    # Calculate the cosine similarity between the query and each context
    similarities = cosine_similarity([query_embedding], context_embeddings)[0]

    # Get the index of the most similar context
    most_similar_index = np.argmax(similarities)

    # If the highest similarity score is above the threshold, return True
    if similarities[most_similar_index] > 0.3:
        return True
    else:
        return False



def get_answer(contexts, query, model):

    # Convert the contexts and query to embeddings
    context_embeddings = model.encode(contexts)
    query_embedding = model.encode([query])[0]

    # Calculate the cosine similarity between the query and each context
    similarities = cosine_similarity([query_embedding], context_embeddings)[0]

    # Get the index of the most similar context
    most_similar_index = np.argmax(similarities)

    # Return the most similar context as the answer
    return contexts[most_similar_index]
