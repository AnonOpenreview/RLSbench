try: 
    import boto3
except Exception as e: 
    pass
import logging
import os
import re
import tarfile

from wilds.datasets.download_utils import extract_archive

logger = logging.getLogger("label_shift")

def get_remote_client(): 

    AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

    s3 = boto3.client(
        service_name='s3',
        region_name='us-west-2',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    return s3

# Adapted from https://stackoverflow.com/questions/57957585/how-to-check-if-a-particular-directory-exists-in-s3-bucket-using-python-and-boto
def check_remote_existence(path_name, bucket, s3_client):

    path_name = re.sub('/+','/', path_name)

    if path_name.startswith("./"): 
        path_name = path_name[2:]
    # if not dir_name.endswith('/'): 
    #     dir_name = dir_name + "/"
    
    resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=path_name)

    return 'Contents' in resp



def download_dir_and_extract(object, path, bucket, s3_client): 
    
    path = re.sub('/+','/', path)
    object = re.sub('/+','/', object)

    logger.info(f"Getting dataset object {object} from remote bucket {bucket} ... ")

    s3_client.download_file(Bucket=bucket, Key = object, Filename=path)

    logger.info("Done.")

    logger.info("Extracting the dataset ...")

    to_path = os.path.dirname(path)

    with tarfile.open(path) as tar:
        tar.extractall(path=to_path)

    os.remove(path)
    
    logger.info("Done.")


def upload_files(local_dir, bucket, s3_client): 

    local_dir = re.sub('/+','/', local_dir)

    if local_dir.startswith("./"): 
        local_dir = local_dir[2:]

    logger.info(f"Uploading to remote bucket ... ")

    for root,dirs,files in os.walk(local_dir):
            for file in files:
                s3_client.upload_file(os.path.join(root,file),bucket, os.path.join(root,file))


    logger.info("Upload finished succesfully.")
