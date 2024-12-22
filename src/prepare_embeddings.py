import os

from pymilvus import Collection
from dotenv import load_dotenv
from langchain_milvus import Milvus
from mistral import MistralEmbed

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")
model_name = "mistral-embed"
api_url = os.getenv("MISTRAL_API_URL")

embed_model = MistralEmbed(api_key, model_name, api_url)


def init_vectorstore_collection(collection_name):
    vector_store = Milvus(
        embedding_function=embed_model,
        collection_name=collection_name,
        connection_args={"host": "127.0.0.1", "port": "19530"},
        auto_id=True
    )

    collection = Collection(collection_name)
    collection.set_properties({"collection.ttl.seconds": 1800})
    return vector_store


def main(collection_name):
    vector_store = init_vectorstore_collection(collection_name)

    documents = [
        "This is the first document.",
        "This is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]
    vector_store.add_texts(documents)

    query = "This is a query document."
    results = vector_store.similarity_search(query)
    print(results)


async def process_data(deque_data):
    while True:
        if deque_data:
            data = deque_data.popleft()
            embeddings = generate_embeddings(data['content'])
            store_embeddings(np.array(embeddings))

if __name__ == "__main__":
    main()
