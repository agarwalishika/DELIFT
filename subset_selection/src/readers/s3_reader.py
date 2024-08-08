import boto3
from botocore.exceptions import ClientError
from .reader import DataReader

class S3DataReader(DataReader):
    """
    Subclass of DataReader for reading data from Amazon S3.
    """
    def __init__(self, bucket, region_name='us-east', endpoint_url=None, aws_access_key_id=None, aws_secret_access_key=None):
        """
        Initialize the S3DataReader with S3 configuration details.
        """
        super().__init__(bucket)
        self.s3 = boto3.resource('s3',
                                 region_name=region_name,
                                 endpoint_url=endpoint_url,
                                 aws_access_key_id=aws_access_key_id,
                                 aws_secret_access_key=aws_secret_access_key)
        self.bucket = bucket

    def read(self, object_name, local_file_name):
        """
        Download a file from S3 to a local file.
        """
        try:
            self.s3.Bucket(self.bucket).download_file(object_name, local_file_name)
            print(f"{object_name} has been downloaded to {local_file_name}")
        except ClientError as e:
            print(f"Download failed: {e}")
            raise e

    def close(self):
        """
        Clean-up method. Currently, there's nothing to clean up for S3.
        """
        pass
