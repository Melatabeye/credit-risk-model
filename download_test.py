import kagglehub

print("Starting download...")

path = kagglehub.dataset_download("atwine/xente-challenge")

print("Downloaded to:", path)