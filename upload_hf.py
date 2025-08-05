from huggingface_hub import HfApi

api = HfApi(token='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')

# Define folder paths and their corresponding repo IDs
folders_to_upload = [
    {
        "folder_path": "/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_4B/finetuned_model/llm",
        "repo_id": "weihezhai/PaperPrediction-All-4B"
    },
    {
        "folder_path": "/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_4B/finetuned_model/cv",
        "repo_id": "weihezhai/PaperPrediction-All-4B"
    },
    {
        "folder_path": "/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_4B/finetuned_model/rl",
        "repo_id": "weihezhai/PaperPrediction-All-4B"
    },
    {
        "folder_path": "/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_4B/finetuned_model/theory",
        "repo_id": "weihezhai/PaperPrediction-All-4B"
    },
    # Add your 3 additional folder configurations here:
    # {
    #     "folder_path": "/path/to/second/folder",
    #     "repo_id": "weihezhai/second-repo-name"
    # },
    # {
    #     "folder_path": "/path/to/third/folder", 
    #     "repo_id": "weihezhai/third-repo-name"
    # },
    # {
    #     "folder_path": "/path/to/fourth/folder",
    #     "repo_id": "weihezhai/fourth-repo-name"
    # }
]

for i, config in enumerate(folders_to_upload, 1):
    print(f"Uploading folder {i}/{len(folders_to_upload)}: {config['folder_path']}")
    try:
        api.upload_folder(
            folder_path=config["folder_path"],
            repo_id=config["repo_id"],
            repo_type="model",
        )
        print(f"✓ Successfully uploaded {config['repo_id']}")
    except Exception as e:
        print(f"✗ Error uploading {config['repo_id']}: {e}")

