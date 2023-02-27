import qdrant_client.http.exceptions
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter


def create_colletion():
    client = QdrantClient(host="localhost", port=6333)

    try:
        client.get_collection(collection_name="products")
    except qdrant_client.http.exceptions.UnexpectedResponse:
        client.recreate_collection(
            collection_name="products",
            vectors_config={
                "image_emb": VectorParams(size=128, distance=Distance.COSINE),
            }
        )


if __name__ == '__main__':
    create_colletion()
