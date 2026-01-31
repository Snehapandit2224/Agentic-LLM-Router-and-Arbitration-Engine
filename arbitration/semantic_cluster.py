from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def cluster_claims(claims, embed_fn, threshold=0.8):
    """
    Groups semantically similar claims.

    claims: list[str]
    embed_fn: function that converts list[str] → embeddings
    threshold: cosine similarity threshold
    """

    embeddings = embed_fn(claims)
    similarity = cosine_similarity(embeddings)

    clusters = []
    visited = set()

    for i, claim in enumerate(claims):
        if i in visited:
            continue

        cluster = [claim]
        visited.add(i)

        for j in range(i + 1, len(claims)):
            if similarity[i][j] >= threshold:
                cluster.append(claims[j])
                visited.add(j)

        clusters.append(cluster)

    return clusters
