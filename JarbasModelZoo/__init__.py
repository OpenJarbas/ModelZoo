import logging
import pickle
from os import listdir
from os.path import join, isfile, dirname

import requests
from xdg import BaseDirectory as XDG

LOG = logging.getLogger("JarbasModelZoo")
LOG.setLevel("INFO")

MODEL_IDS = [f.replace(".json", "")
             for f in listdir(join(dirname(__file__), "res"))]

_BASE_URL = "https://github.com/OpenJarbas/ModelZoo/releases/download"
_VERSION = "0.2.0a2"
MODEL2URL = {model_id: _BASE_URL + f"/{_VERSION}/{model_id}"
             for model_id in MODEL_IDS}


def download(model_id, force=False):
    if model_id in MODEL_IDS:
        url = MODEL2URL[model_id]
    else:
        raise ValueError("invalid model_id")
    path = join(XDG.save_data_path("JarbasModelZoo"), model_id + ".pkl")
    if isfile(path) and not force:
        LOG.info("Already downloaded " + model_id)
        return
    LOG.info("downloading " + model_id)
    LOG.info(url)
    LOG.info("this might take a while...")
    with open(path, "wb") as f:
        f.write(requests.get(url).content)


def load_model(model_id):
    if isfile(model_id):
        path = model_id
    else:
        path = join(XDG.save_data_path("JarbasModelZoo"), model_id + ".pkl")
        if not isfile(path):
            download(model_id)
    LOG.debug("loading: " + path)
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
