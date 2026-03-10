import os
from huggingface_hub import snapshot_download

# 显式指定（双重保险）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

save_dir = "/vla/users/niejunnan/codebase/serl_torch/pretrained_models"
os.makedirs(save_dir, exist_ok=True)

models = ["microsoft/resnet-18", "microsoft/resnet-50"]

for model_id in models:
    local_dir = os.path.join(save_dir, model_id.replace("/", "--"))
    print(f"Starting download: {model_id} to {local_dir}")
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            ignore_patterns=["*.h5", "flax_model*", "*.ot", "*.msgpack"],
            resume_download=True,      # 断点续传
            max_workers=4,             # 多线程加速
            endpoint="https://hf-mirror.com" # 强制指定 endpoint
        )
        print(f"Successfully downloaded: {model_id}")
    except Exception as e:
        print(f"Failed to download {model_id}. Error: {e}")