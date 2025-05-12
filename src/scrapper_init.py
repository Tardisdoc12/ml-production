################################################################################
# filename: scrapper_init.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 12/05,2025
################################################################################

import json
import os
from pathlib import Path

import requests

################################################################################


def load_config(path="config/scraper.json"):
    return json.load(open(path))


################################################################################


def main():
    cfg = load_config()
    os.makedirs("data/raw", exist_ok=True)
    params = cfg.get("params", {})
    resp = requests.get(cfg["endpoint"], params=params)
    for i, item in enumerate(resp.json()[:50]):
        fname = Path("data/raw") / f"{i}.json"
        fname.write_text(json.dumps(item))
    print("50 fichiers téléchargés dans data/raw/")


################################################################################

if __name__ == "__main__":
    main()

################################################################################
# End of File
################################################################################
