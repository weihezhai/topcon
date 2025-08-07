from huggingface_hub import HfApi, HfFolder
from huggingface_hub.utils import RepositoryNotFoundError

api = HfApi(token='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')

# Define folder paths and their corresponding repo IDs
folders_to_upload = [
    {
        "folder_path": "/mnt/parscratch/users/acr24wz/etu/topcon/Llama-3.2-1B/finetuned_model/llm",
        "repo_id": "PaperPred/PaperPrediction-LLM-1B"
    }
    # {
    #     "folder_path": "/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_1d7B/finetuned_model/llm",
    #     "repo_id": "weihezhai/PaperPrediction-LLM-Qwen3-1D7B"
    # },
    # {
    #     "folder_path": "/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_1d7B/finetuned_model/cv",
    #     "repo_id": "weihezhai/PaperPrediction-CV-Qwen3-1D7B"
    # },
    # {
    #     "folder_path": "/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_1d7B/finetuned_model/rl",
    #     "repo_id": "weihezhai/PaperPrediction-RL-Qwen3-1D7B"
    # },
    # {
    #     "folder_path": "/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_1d7B/finetuned_model/theory",
    #     "repo_id": "weihezhai/PaperPrediction-Th-Qwen3-1D7B"
    # },
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
    print(f"\nProcessing {i}/{len(folders_to_upload)}: {config['repo_id']}")
    
    # Check if repo exists, create if it doesn't
    try:
        api.repo_info(repo_id=config["repo_id"], repo_type="model")
        print(f"  Repository exists: {config['repo_id']}")
    except RepositoryNotFoundError:
        print(f"  Creating repository: {config['repo_id']}")
        api.create_repo(repo_id=config["repo_id"], repo_type="model", private=False)
    
    # Upload folder
    print(f"  Uploading folder: {config['folder_path']}")
    try:
        api.upload_folder(
            folder_path=config["folder_path"],
            repo_id=config["repo_id"],
            repo_type="model",
        )
        print(f"  ✓ Successfully uploaded to {config['repo_id']}")
    except Exception as e:
        print(f"  ✗ Error uploading to {config['repo_id']}: {e}")

