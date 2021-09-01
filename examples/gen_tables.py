from os import listdir
from os.path import join, dirname
import json


def generate_NER():
    header= """
|model_id|language|dataset|accuracy|
|:------:|:------:|:------:|:------:|"""

    table = header
    items = []
    for f in listdir(join(dirname(__file__), "../JarbasModelZoo/res")):
        if not f.endswith(".json"):
            continue
        if "NER" not in f:
            continue

        with open(join(dirname(__file__), "../JarbasModelZoo/res", f)) as fi:
            meta = json.load(fi)
            meta = {k:v.replace("|", "/") if isinstance(v, str) else v
                    for k, v in meta.items()}
            items += [meta]


    items = sorted(items, key=lambda k: k["corpus"])
    for meta in items:
        if meta.get("corpus_homepage"):
            meta["corpus"] = f"""[{meta["corpus"]}]({meta["corpus_homepage"]})"""
        meta["required_packages"] = " + ".join([p for p in meta.get("required_packages", [])])
        table += "\n" + f"""|{meta["model_id"]}|{meta["lang"]}|{meta["corpus"]}| {str(meta.get("accuracy", 0))[:5]}% |"""
    return table


def generate_POSTAG():
    header= """
|model_id|language|dataset|tagset|accuracy|
|:------:|:------:|:------:|:------:|:------:|"""

    table = header
    items = []
    for f in listdir(join(dirname(__file__), "../JarbasModelZoo/res")):
        if not f.endswith("tagger.json") or f.endswith("chunk_tagger.json"):
            continue

        with open(join(dirname(__file__), "../JarbasModelZoo/res", f)) as fi:
            meta = json.load(fi)
            meta = {k:v.replace("|", "/") if isinstance(v, str) else v
                    for k, v in meta.items()}
            items += [meta]


    items = sorted(items, key=lambda k: k["corpus"])
    items = sorted(items, key=lambda k: k["tagset"], reverse=True)
    for meta in items:
        if meta.get("corpus_homepage"):
            meta["corpus"] = f"""[{meta["corpus"]}]({meta["corpus_homepage"]})"""
        if meta.get("tagset_homepage"):
            meta["tagset"] = f"""[{meta["tagset"]}]({meta["tagset_homepage"]})"""
        meta["required_packages"] = " + ".join([p for p in meta.get("required_packages", ["nltk"])])
        table += "\n" + f"""|{meta["model_id"]}|{meta["lang"]}|{meta["corpus"]}|{meta["tagset"]}| {str(meta.get("accuracy", 0))[:5]}% |"""
    return table


def generate_CHUNKERS():
    header= """
|model_id|language|dataset|tagset|accuracy|
|:------:|:------:|:------:|:------:|:------:|"""

    table = header
    items = []
    for f in listdir(join(dirname(__file__), "../JarbasModelZoo/res")):
        if not f.endswith("chunk_tagger.json"):
            continue

        with open(join(dirname(__file__), "../JarbasModelZoo/res", f)) as fi:
            meta = json.load(fi)
            meta = {k:v.replace("|", "/") if isinstance(v, str) else v
                    for k, v in meta.items()}
            items += [meta]


    items = sorted(items, key=lambda k: k["corpus"])
    items = sorted(items, key=lambda k: k["tagset"], reverse=True)
    for meta in items:
        if meta.get("corpus_homepage"):
            meta["corpus"] = f"""[{meta["corpus"]}]({meta["corpus_homepage"]})"""
        if meta.get("tagset_homepage"):
            meta["tagset"] = f"""[{meta["tagset"]}]({meta["tagset_homepage"]})"""
        meta["required_packages"] = " + ".join([p for p in meta.get("required_packages", ["nltk"])])
        table += "\n" + f"""|{meta["model_id"]}|{meta["lang"]}|{meta["corpus"]}|{meta["tagset"]}| {str(meta.get("accuracy", 0))[:5]}% |"""
    return table


print("### NER")
print(generate_NER())

print("\n### POSTAG")
print(generate_POSTAG())

print("\n### CHUNKING")
print(generate_CHUNKERS())