"""Download the Sportec Open DFL dataset (7 Bundesliga matches, ~2.6 GB, CC BY 4.0).

    python -m possession.download_dfl                 # -> ../dfl_data next to this repo
    DFL_DATA_DIR=D:/dfl python -m possession.download_dfl

Source: Bassek et al. (2025), https://doi.org/10.6084/m9.figshare.28196177
Credit: Deutsche Fussball Liga (DFL). Re-running skips files that are already complete.
"""

import json
import os
import urllib.request

from .common import DFL_DIR

ARTICLE = "https://api.figshare.com/v2/articles/28196177"


def main(out=DFL_DIR):
    os.makedirs(out, exist_ok=True)
    files = json.load(urllib.request.urlopen(ARTICLE, timeout=60))["files"]
    for f in sorted(files, key=lambda f: f["size"]):
        path = os.path.join(out, f["name"])
        if os.path.exists(path) and os.path.getsize(path) == f["size"]:
            print("skip", f["name"], flush=True)
            continue
        print("get ", f["name"], "{:.1f} MB".format(f["size"] / 1e6), flush=True)
        urllib.request.urlretrieve(f["download_url"], path + ".part")
        os.replace(path + ".part", path)
    print("done ->", out)


if __name__ == "__main__":
    main()
