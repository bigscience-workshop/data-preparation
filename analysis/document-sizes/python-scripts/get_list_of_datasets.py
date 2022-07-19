from huggingface_hub import HfApi

hf_api = HfApi()
author = "bigscience-catalogue-lm-data"
list_datasets = hf_api.list_datasets(author=author, use_auth_token=True)
print("Number of datasets", len(list_datasets))

with open("list_ds.txt", "w") as f:
    for item in [ds_info.id[len(f"{author}/") :] for ds_info in list_datasets]:
        f.write(f'"{item}"\n')

print(dir(list_datasets[0]))
