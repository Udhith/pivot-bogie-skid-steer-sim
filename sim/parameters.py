import yaml, pathlib
def load(path=None):
    p = pathlib.Path(path) if path else pathlib.Path(__file__).resolve().parent.parent / "params.yaml"
    raw = yaml.safe_load(open(p))
    return {k: v["value"] for k, v in raw.items()}, {k: v["src"] for k, v in raw.items()}
