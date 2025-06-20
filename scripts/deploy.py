# scripts/deploy.py

from huggingface_hub import HfApi, HfFolder, create_repo
import os

def deploy_to_hf():
    # Le token est lu depuis les secrets GitHub
    hf_token = os.environ.get("HF_API_KEY")
    if not hf_token:
        raise ValueError("Hugging Face API token not found. Please set the HF_API_KEY secret.")

    # S'authentifier
    HfFolder.save_token(hf_token)
    
    # Nom du repo sur le Hub Hugging Face
    # Il est bon de le préfixer avec votre nom d'utilisateur pour éviter les conflits
    repo_name = "NchourupouoM/heart-disease-cicd-project"

    # Créer le repo s'il n'existe pas
    try:
        create_repo(repo_id=repo_name, repo_type="model", exist_ok=True)
        print(f"Repo '{repo_name}' created or already exists.")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    api = HfApi()

    # Uploader le modèle
    try:
        api.upload_file(
            path_or_fileobj="models/logistic_regression_model.joblib",
            path_in_repo="logistic_regression_model.joblib",
            repo_id=repo_name,
            repo_type="model",
        )
        print(f"Model file uploaded to '{repo_name}'.")

        # Optionnel mais recommandé : Uploader un fichier README.md (Model Card)
        readme_content = f"""
            ---
            license: mit
            ---
            # Heart Disease Classifier
            This repository contains a simple logistic regression model to predict heart disease.
            The model was trained on the UCI Heart Disease dataset.
        """
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="model",
        )
        print(f"README.md uploaded to '{repo_name}'.")

    except Exception as e:
        print(f"Error uploading to Hub: {e}")

if __name__ == "__main__":
    deploy_to_hf()