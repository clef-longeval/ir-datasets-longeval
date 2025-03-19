from ir_datasets import main_cli as irds_main_cli
from ir_datasets_longeval.longeval_web import register as register_longeval_web
from ir_datasets_longeval.longeval_sci import register as register_longeval_sci


def register(sub_collections=None) -> None:
    dataset = sub_collections.split("/")[0]
    if dataset == "longeval-sci":
        register_longeval_sci()
    elif dataset == "longeval-web":
        register_longeval_web()
    else:
        register_longeval_web()
        register_longeval_sci()


def main_cli() -> None:
    register()
    irds_main_cli()
