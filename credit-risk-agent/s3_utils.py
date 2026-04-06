import boto3
import os
import io  
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")  # Make sure env var name matches

def upload_file_to_s3(file_obj, customer_id):
    # 1. Check Config
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY or not AWS_BUCKET_NAME:
        print("❌ S3 Config Error: Missing AWS credentials")
        return None

    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )
    
    try:
        # 2. Create a "Safe Copy" for S3
        # We read the file into memory so Boto3 can close its own copy 
        # without affecting the original file_obj needed for Pinecone.
        file_obj.seek(0)
        file_content = file_obj.read()
        data_stream = io.BytesIO(file_content)
        
        file_key = f"financial_documents/{customer_id}/{file_obj.name}"
        
        # 3. Upload the COPY
        s3.upload_fileobj(data_stream, AWS_BUCKET_NAME, file_key)
        
        # 4. Reset the ORIGINAL file for the next function (Parsing)
        file_obj.seek(0)
        
        return file_key
        
    except Exception as e:
        print(f"❌ S3 Upload Error: {e}")
        return None
