from google.cloud import storage
import os

# Code for accessing data from Google Cloud Bucket
def download_dataset(bucket_name, source_folder, destination_folder):
    # Set up the environment variable for authentication
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/alejandro/Desktop/keys/cap5415-442520-c7b56ba5aa5b.json"

    # Initialize the client
    client = storage.Client()

    # Specify your bucket name
    # bucket = client.bucket(bucket_name)
    bucket = client.get_bucket(bucket_name)

    # List all files in the bucket
    blobs = bucket.list_blobs(prefix=source_folder)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for blob in blobs:
        file_path = os.path.join(destination_folder, os.path.basename(blob.name))
        blob.download_to_filename(file_path)
        print(f"Downloaded {blob.name} to {file_path}")


# Download dataset
bucket_name = "nih-chest-xray-project"
source_folder = "images/"
destination_folder = "./data/raw"
download_dataset(bucket_name, source_folder, destination_folder)
