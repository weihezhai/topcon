from huggingface_hub import HfApi

api = HfApi(token='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')
api.upload_folder(
    folder_path="/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_4B/finetuned_model/all_four_domain",
    repo_id="weihezhai/PaperPrediction-All-4B",
    repo_type="model",
)