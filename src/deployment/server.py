import litserve as ls
from src.deployment.online.api import TitanicInferenceAPI

if __name__ == "__main__":
    api = TitanicInferenceAPI()
    server = ls.LitServer(
        api,
        accelerator="cpu"
    )
    server.run(port=8000, generate_client_file=False)