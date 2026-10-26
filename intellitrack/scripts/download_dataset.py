import kagglehub

# Download latest version
path = kagglehub.dataset_download("foryolotrain1/visdrone2019-mot")

print("Path to dataset files:", path)