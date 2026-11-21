from huggingface_hub import HfApi

api = HfApi()
repo = "bgeary/al-valuation-model"
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)

files = [
    r"models\rf_pipeline.joblib",
    r"models\model_columns.pkl",
    r"notebooks\models\area_map.joblib",
]
for f in files:
    api.upload_file(path_or_fileobj=f, path_in_repo=f.split("\\")[-1], repo_id=repo)
    print("uploaded", f)