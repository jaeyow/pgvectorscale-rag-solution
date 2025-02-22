import json
import boto3


class TitanEmbeddings(object):
    accept = "application/json"
    content_type = "application/json"

    def __init__(self, model_id="amazon.titan-embed-text-v2:0", **kwargs):
        self.aws_access_key_id = kwargs.get("aws_access_key_id")
        self.aws_secret_access_key = kwargs.get("aws_secret_access_key")
        self.aws_session_token = kwargs.get("aws_session_token")
        self.region_name = kwargs.get("region_name")

        self.bedrock = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            region_name=self.region_name,
        )

        self.model_id = model_id

    def __call__(self, text, dimensions=1024, normalize=True):
        """
        Returns Titan Embeddings
        Args:
            text (str): text to embed
            dimensions (int): Number of output dimensions.
            normalize (bool): Whether to return the normalized embedding or not.
        Return:
            List[float]: Embedding

        """
        body = json.dumps(
            {"inputText": text, "dimensions": dimensions, "normalize": normalize}
        )
        response = self.bedrock.invoke_model(
            body=body,
            modelId=self.model_id,
            accept=self.accept,
            contentType=self.content_type,
        )
        response_body = json.loads(response.get("body").read())
        return response_body["embedding"]
