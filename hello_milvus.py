from pymilvus import connections, MilvusClient, FieldSchema, CollectionSchema, DataType, Collection, model
import numpy as np
import random

# Function to initialize Milvus Client and set up the database
def initialize_milvus():
    # Connect to the Milvus server
    connections.connect("default", host="34.81.214.153", port="19530")
    
    # Create a local Milvus vector database
    client = MilvusClient("milvus_demo.db")

    # Define a schema for the collection
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    schema = CollectionSchema(fields, "Demo collection for vector search")

    # Create a collection
    collection_name = "demo_collection"
    collection = Collection(name=collection_name, schema=schema)

    return client, collection

# Function to generate data and insert it into the collection
def prepare_and_insert_data(client, collection):
    # Represent text with vectors using a pre-trained embedding model
    embedding_fn = model.DefaultEmbeddingFunction()

    # Text strings to search from
    docs = [
        "Artificial intelligence was founded as an academic discipline in 1956.",
        "Alan Turing was the first person to conduct substantial research in AI.",
        "Born in Maida Vale, London, Turing was raised in southern England.",
    ]

    try:
        vectors = embedding_fn.encode_documents(docs)
    except:
        # Use fake representation with random vectors if model download fails
        vectors = [ [ random.uniform(-1, 1) for _ in range(768) ] for _ in docs ]

    data = [ {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"} for i in range(len(vectors)) ]

    # Insert data into the collection
    client.insert(
        collection_name="demo_collection",
        data=data
    )

    return vectors, docs

# Function to perform a vector search
def perform_vector_search(client, vectors):
    # Perform a vector search
    search_vectors = vectors[:1]  # Use the first vector for search
    res = client.search(
        collection_name="demo_collection",  # target collection
        data=search_vectors,                # query vectors
        limit=2,                            # number of returned entities
        output_fields=["text", "subject"],  # specifies fields to be returned
    )

    print("Search results:")
    for result in res:
        print(result)

# Function to perform additional operations like metadata filtering, query, and delete
def additional_operations(client):
    # Insert more documents with different subjects
    embedding_fn = model.DefaultEmbeddingFunction()
    docs = [
        "Machine learning has been used for drug design.",
        "Computational synthesis with AI algorithms predicts molecular properties.",
        "DDR1 is involved in cancers and fibrosis.",
    ]
    vectors = embedding_fn.encode_documents(docs)
    data = [ {"id": 3+i, "vector": vectors[i], "text": docs[i], "subject": "biology"} for i in range(len(vectors))]

    client.insert(
        collection_name="demo_collection",
        data=data
    )

    # Perform a vector search with metadata filtering
    res = client.search(
        collection_name="demo_collection",
        data=embedding_fn.encode_queries([ "tell me AI related information" ]),
        filter="subject == 'biology'",
        limit=2,
        output_fields=["text", "subject"],
    )

    print("Filtered search results:")
    for result in res:
        print(result)

    # Query by filter
    res = client.query(
        collection_name="demo_collection",
        filter="subject == 'history'",
        output_fields=["text", "subject"],
    )
    print("Query by filter results:", res)

    # Query by primary key
    res = client.query(
        collection_name="demo_collection",
        ids=[0, 2],
        output_fields=["vector", "text", "subject"]
    )
    print("Query by primary key results:", res)

    # Delete entities by primary key
    res = client.delete(
        collection_name="demo_collection",
        ids=[0, 2]
    )
    print("Delete by primary key results:", res)

    # Delete entities by filter
    res = client.delete(
        collection_name="demo_collection",
        filter="subject == 'biology'",
    )
    print("Delete by filter results:", res)

    # Drop the collection
    client.drop_collection(
        collection_name="demo_collection"
    )
    print("Collection dropped.")

if __name__ == "__main__":
    client, collection = initialize_milvus()
    vectors, docs = prepare_and_insert_data(client, collection)
    perform_vector_search(client, vectors)
    additional_operations(client)
