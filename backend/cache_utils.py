from redis import Redis
from redis.exceptions import ConnectionError
from redisvl.query import RangeQuery
from redisvl.index import SearchIndex
from redisvl.utils.vectorize import OpenAITextVectorizer
import os
import boto3
from botocore.exceptions import ClientError

class CacheServer:
    index = None
    blob_storage = None

    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION', 'us-west-1')
    S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'dreamscapeassetbucket')

    def init(self, redis_host: str = 'localhost', redis_port: str = '6379') -> SearchIndex:
        # INSTANTIATE REDIS
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
        index.connect(f"redis+cluster://{redis_host}:{redis_port}/0")
        index.create(overwrite=True)

        self.index = index

        # INSTANTIATE BLOB STORAGE
        self.blob_storage = boto3.client('s3')

    def getEmbedding(objectName):
        api_key = os.environ.get("OPENAI_API_KEY")
        oai = OpenAITextVectorizer(
            model="text-embedding-ada-002",
            api_config={"api_key": api_key},
        )
        embedding = oai.embed(objectName)
        return embedding

    def get(self, embedding):
        # Search for matching objects
        v = RangeQuery(embedding, "user_object_index", return_fields=["url"], num_results=1)
        results = self.index.query(v)

        # return top result's url
        return results[0]["url"] if len(results) > 0 else False
    
    def post(self, obj_file_path, embedding):
        # Upload file to S3
        with open(obj_file_path, "rb") as f:
            try:
                self.blob_storage.upload_fileobj(f, self.S3_BUCKET_NAME, obj_file_path)
            except ClientError as e:
                print(f"Error uploading file to S3: {e}")
                return None

        # Generate the URL for the uploaded object
        try:
            url = self.blob_storage.generate_presigned_url('get_object',
                                                Params={'Bucket': bucket_name,
                                                        'Key': object_key},
                                                ExpiresIn=86400)  # URL expires in 1 hour
        except ClientError as e:
            print(f"Error generating presigned URL: {e}")
            return None

        # Add mapping to vector db
        data = [{
                "user_object_index": embedding,
                "url": url
        }]
        self.index.load(data)