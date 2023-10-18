import pandas as pd
import numpy as np
from google.cloud import storage
import os, shutil
from IPython.display import clear_output
import time

class cloud_data:
    def __init__():
        pass
    
    def google_cloud(file_data=[],
                     bucket_name='', 
                     credentials_path='',
                     folders_prefix=[],
                     local_folder='',
                     path_cloud_name=''):
        storage_client=storage.Client.from_service_account_json(credentials_path)
        
        gt_bucket=storage_client.get_bucket(bucket_name)
        
        #Setting the location of storage
        blobs=gt_bucket.blob(path_cloud_name)
        
        for prefix in folders_prefix:
            str_local=prefix.split('/')[0]
            new_dir=os.path.join(os.getcwd(),local_folder)
            os.makedirs(new_dir,exist_ok=True)
            get_objects=gt_bucket.list_blobs(prefix=prefix)

            ##Download each file
            for obj in get_objects:
                
                #setting up the location of your local data
                local=os.path.join(new_dir,str_local)
                os.makedirs(local,exist_ok=True)
                filename=obj.name.replace('/','_')
                print(f'This is {filename} from GCP Bucket')
                print(f"Sending and Checking file {filename} to the local {local}")
                #Download the object to local data
                if (filename in os.listdir(local)):
                    print(f"Your {filename} had been downloaded before")
                    clear_output(wait=True)
                    time.sleep(5)
                    continue
                else:
                    obj.download_to_filename(local+'/'+filename)
                    print(f'Downloaded: {obj.name} to {local_folder} is Success')
                    clear_output(wait=True)
                    time.sleep(5)

            