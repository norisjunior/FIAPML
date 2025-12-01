# >>> COLOQUE ESTAS 2 LINHAS NO TOPO DO ARQUIVO, ANTES DO IMPORT KAGGLE <<<
import os, pathlib
os.environ["KAGGLE_CONFIG_DIR"] = str(pathlib.Path(__file__).parent)  # onde está o kaggle.json

from kaggle.api.kaggle_api_extended import KaggleApi

import shutil, stat, sys
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

# Dataset alvo (Kaggle -> Datasets -> mlippo/car-accidents-in-brazil-2017-2023)
DATASET  = "mlippo/car-accidents-in-brazil-2017-2023"
DEST_DIR = "brazil_car_accidents"
CSV_NAME = "accidents_2017_to_2023_portugues.csv"  # ajuste se necessário

def ensure_kaggle_credentials():
    """
    Procura credenciais em:
      1) ~/.kaggle/kaggle.json (já pronto)
      2) kaggle.json ao lado do script (copia p/ ~/.kaggle/)
      3) variáveis de ambiente KAGGLE_USERNAME e KAGGLE_KEY (gera kaggle.json)
    """
    home = os.path.expanduser("~")
    kdir = os.path.join(home, ".kaggle")
    kfile = os.path.join(kdir, "kaggle.json")

    if os.path.exists(kfile):
        return  # já configurado

    # 2) kaggle.json ao lado do script
    local_json = os.path.join(os.path.dirname(__file__), "kaggle.json")
    if os.path.exists(local_json):
        os.makedirs(kdir, exist_ok=True)
        shutil.copy(local_json, kfile)
        if os.name != "nt":  # permissões tipo 600 em Unix
            os.chmod(kfile, stat.S_IRUSR | stat.S_IWUSR)
        print("kaggle.json copiado para ~/.kaggle/")
        return

    # 3) variáveis de ambiente
    user = os.getenv("KAGGLE_USERNAME")
    key  = os.getenv("KAGGLE_KEY")
    if user and key:
        os.makedirs(kdir, exist_ok=True)
        with open(kfile, "w", encoding="utf-8") as f:
            f.write(f'{{"username":"{user}","key":"{key}"}}')
        if os.name != "nt":
            os.chmod(kfile, stat.S_IRUSR | stat.S_IWUSR)
        print("kaggle.json criado em ~/.kaggle/ a partir de variáveis de ambiente.")
        return

    raise RuntimeError(
        "Credenciais não encontradas. Coloque 'kaggle.json' ao lado do script "
        "OU defina KAGGLE_USERNAME e KAGGLE_KEY no ambiente."
    )

def main():
    ensure_kaggle_credentials()

    api = KaggleApi()
    api.authenticate()

    os.makedirs(DEST_DIR, exist_ok=True)
    print("Baixando e extraindo…")
    api.dataset_download_files(DATASET, path=DEST_DIR, unzip=True)

    print("Arquivos baixados:", os.listdir(DEST_DIR))

    csv_path = os.path.join(DEST_DIR, CSV_NAME)
    if not os.path.exists(csv_path):
        # fallback: abre o primeiro CSV encontrado
        csvs = [f for f in os.listdir(DEST_DIR) if f.lower().endswith(".csv")]
        if not csvs:
            print("Nenhum CSV encontrado na pasta de destino.")
            sys.exit(0)
        csv_path = os.path.join(DEST_DIR, csvs[0])

    df = pd.read_csv(csv_path)
    print(df.head())
    print(df.info())

if __name__ == "__main__":
    main()
