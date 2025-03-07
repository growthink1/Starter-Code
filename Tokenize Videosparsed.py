#!/usr/bin/env python
# coding: utf-8

# In[86]:


import sagemaker, boto3
import pandas as pd 
from io import StringIO
import json
import requests
import botocore
from IPython.display import Video, display
import os
import time
import concurrent.futures
import pandas as pd


# ## Set up resources and source/destination bucket names

# In[20]:


role = sagemaker.get_execution_role()
### GET DATA FROM S3 ### 
s3 = boto3.client('s3')
transfer_mgr = boto3.s3.transfer.TransferManager(s3)
source_bucket = "proj-506-general-wh-data"
li_video_file = "linkedin_data_scratch/video_download_links.csv"
response = s3.get_object(Bucket=source_bucket, Key=li_video_file)
body = response['Body']
csv_string = body.read().decode('utf-8')
video_download_links_df = pd.read_csv(StringIO(csv_string))
video_download_links_df = video_download_links_df[["video_urn","download_link"]].drop_duplicates()
video_download_links_df = video_download_links_df.reset_index().drop(columns=["index"])
video_format = 'mp4'
destination_bucket = 'prj-506-video-data'
destination_folder = "li_videos/"


# In[16]:


download_links = list(video_download_links_df["download_link"])
video_names = list(video_download_links_df["video_urn"].apply(lambda s : s[s.rindex(':')+1:]))
video_names = [f"{video_name}.mp4" for video_name in video_names]
print(download_links[1])
print(video_names[1])


# # The below cell downloads videos to the s3 bucket and does not need to be run!!

# In[23]:


# counter = 0
# for i in range(len(download_links)): 
#     video_url = download_links[i]
#     video_name = video_names[i]
#     if video_url != "urn not in response":
#         response = requests.get(video_url)

#     if response.status_code == 200 :
#         video_filename = destination_folder + video_name
#         s3.put_object(Bucket=destination_bucket,
#                          Key=video_filename,
#                          Body=response.content)  

#     else: 
#         print("this video is not downloaded:", video_name)
#         video_not_download.append(video_name)
#     counter += 1
#     print(counter)


# In[ ]:


li_vids = [obj.key for obj in bucket.objects.filter(Prefix = "li_videos/")][1:]
rekognition_client = boto3.client('rekognition', region_name="us-east-1")
counter = 0
s3 = boto3.resource('s3')
bucket = s3.Bucket('prj-506-video-data')

def detect_objects(fname):
    global counter
    s3_bucket_name = 'prj-506-video-data'
    s3_video_key = fname
    
    video_s3_object = {'S3Object': {'Bucket': s3_bucket_name, 'Name': s3_video_key}}
    
    try:
        start_response = rekognition_client.start_label_detection(
        Video=video_s3_object,
        MinConfidence=80,
        JobTag = 'label-detection-job'
    )
    except rekognition_client.exceptions.LimitExceededException as e:
        print(f'Hit limit for starting label detection job for {fname}. Waiting for 1 second and trying again.')
        time.sleep(1)
        return detect_objects(fname)
    job_id = start_response['JobId']
    print(f'Started job {job_id} for {fname}')
    
    while True:
        label_response = rekognition_client.get_label_detection(JobId=job_id)
        status = label_response['JobStatus']
        if status == 'SUCCEEDED':
            break
        elif status == 'FAILED':
            print(f'Job {job_id} for {fname} failed!')
            return ""
        else:
            print(f'Job {job_id} for {fname} status: {status}. Waiting...')
            time.sleep(5)

    label_response = rekognition_client.get_label_detection(JobId=start_response["JobId"])
    
    try:
        objects = {label["Label"]["Name"] for label in label_response["Labels"]}
    except KeyError:
        objects = set()
    except:
        raise
    counter += 1
    print(counter)
    return " ".join(objects)
        


# In[100]:


def split_list(lst, chunk_size):
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

li_vids_chunks = split_list(li_vids, 10)
#object_detection_str_list = list(map(detect_objects, li_vids))


# In[ ]:


counter = 1
for chunk in li_vids_chunks:
    obj_df = pd.DataFrame()
    object_detection_str_list = list(map(detect_objects, chunk))
    obj_df["video_id"] = chunk
    obj_df["object_detection_str"] = object_detection_str_list
    
    obj_df.to_csv(f"s3://proj-506-general-wh-data/linkedin_data_scratch/video_object_data/obj_chunk{counter}.csv", index=False)
    print(f"finished chunk {counter}")
    counter += 1


# In[84]:


len(li_vids_chunks)


# In[91]:


s3_bucket_name = 'prj-506-video-data'
s3_video_key = 'li_videos/C4D05AQEFC3KsRwK1UA.mp4'

# Replace with the AWS Region you're using
region_name = 'us-east-1'

# Initialize the Amazon Rekognition client
rekognition_client = boto3.client('rekognition', region_name=region_name)

# Specify the S3 bucket and video key
video_s3_object = {'S3Object': {'Bucket': s3_bucket_name, 'Name': s3_video_key}}

# Start content moderation for the video
response = rekognition_client.start_label_detection(
    Video=video_s3_object,
    MinConfidence=80,
    JobTag = 'label-detection-job'
)

# Print the response
print(response)


# In[92]:


label_response = rekognition_client.get_label_detection(JobId=response["JobId"])


# In[99]:


label_response



# In[98]:


s3 = boto3.resource('s3')
bucket = s3.Bucket('prj-506-video-data')

li_vids = [obj.key for obj in bucket.objects.filter(Prefix = "li_videos/")][1:]
print(li_vids)


# In[48]:


" ".join(set())


# In[ ]:




