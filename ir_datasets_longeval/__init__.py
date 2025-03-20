from ir_datasets import main_cli as irds_main_cli, registry as irds_registry

from ir_datasets_longeval.longeval_sci import register as register_longeval_sci, LongEvalSciDataset
from ir_datasets_longeval.longeval_web import register as register_longeval_web
from typing import Union
from pathlib import Path


def load(longeval_ir_dataset: Union[str, Path]):
    """_summary_

    Args:
        longeval_ir_dataset (Union[str, Path]): _description_
    """
    register_longeval_sci()
    exists_locally = longeval_ir_dataset and Path(longeval_ir_dataset).exists() and Path(longeval_ir_dataset).is_dir()
    exists_in_irds = longeval_ir_dataset in irds_registry and irds_registry[longeval_ir_dataset]

    if exists_locally and exists_in_irds:
        raise ValueError('foo')

    if exists_locally:
        return LongEvalSciDataset(Path(longeval_ir_dataset))

    if exists_in_irds:
        return irds_registry[longeval_ir_dataset]

    raise ValueError('foo')

def register(dataset=None) -> None:
    if dataset:
        dataset = dataset.split("/")[0]
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
