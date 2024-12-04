
from huggingface_hub import notebook_login

notebook_login()

# Make sure that you add acess to this repo by editing your access token. 

from datasets import load_dataset

hf_dataset_identifier = "engrodawood/segment_rand"
ds = load_dataset(hf_dataset_identifier)

print(ds)