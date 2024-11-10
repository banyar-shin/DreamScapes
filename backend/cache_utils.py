from redis import Redis
from redis.exceptions import ConnectionError
from redisvl.query import RangeQuery
from redisvl.index import SearchIndex

class CacheServer:
    index = None
    blob_storage = None

    def init(self, redis_host: str = 'localhost', redis_port: str = '6379') -> SearchIndex:
        # Initialize Redis client
        redis_client = Redis(host=redis_host, port=int(redis_port), db=0)

        # Verify Redis connection
        try:
            redis_client.ping()
            print("Connected to Redis")
        except ConnectionError:
            print("Failed to connect to Redis")
            exit(1)

        # Define Redis schema
        schema = {
            "index": {
                "name": "user_object_index",
                "prefix": "user_voice_object_description:",
            },
            "fields": [
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "datatype": "float32",
                        "dims": 512,
                        "distance_metric": "COSINE",  # or "L2" for Euclidean distance
                        "algorithm": "HNSW"
                    },
                },
                {
                    "name": "url",
                    "type": "text",
                }
            ]
        }

        # Create SearchIndex from schema
        index = SearchIndex.from_dict(schema)

        # Set Redis client for the index
        index.set_client(redis_client)

        # Connect and create the index
        index.connect(f"redis://{redis_host}:{redis_port}")
        index.create(overwrite=True)

        self.index = index

    def getEmbedding(objectName):
        # Get embedding
        embedding = ""
        return embedding

    def get(self, embedding):
        # Search for matching objects
        v = RangeQuery(embedding, "user_object_index", return_fields=["url"], num_results=1)
        results = self.index.query(v)

        # return top result's url
        return results[0]["url"] if len(results) > 0 else False
    
    def post(self, embedding, url):
        # add object to blob storage
        #TODO: add object to blob storage
        url = "" # TODO: get url of the object from blob storage

        # Add mapping to vector db
        data = [{
                "user_object_index": embedding,
                "url": url
        }]
        self.index.load(data)

