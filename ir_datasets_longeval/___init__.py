from pathlib import Path
from typing import Union
import json 

from ir_datasets import main_cli as irds_main_cli
from ir_datasets import registry as irds_registry

from ir_datasets_longeval.longeval_sci import LongEvalSciDataset
from ir_datasets_longeval.longeval_sci import register as register_longeval_sci
from ir_datasets_longeval.longeval_web import LongEvalWebDataset
from ir_datasets_longeval.longeval_web import register as register_longeval_web

def read_property_from_metadata(base_path, property):
    return json.load(open(base_path / "metadata.json", "r"))[property]

def load(longeval_ir_dataset: Union[str, Path]):
    """Load an LongEval ir_dataset. Can point to an official ID of an LongEval dataset or a local directory of the same structure.

    Args:
        longeval_ir_dataset (Union[str, Path]): the ID of an LongEval ir_dataset or a local path.
    """
    if longeval_ir_dataset is None:
        raise ValueError("Please pass either a string or a Path.")

    if longeval_ir_dataset.startswith("longeval-sci"):
        register_longeval_sci()
    if longeval_ir_dataset.startswith("longeval-web"):
        register_longeval_web()

    exists_locally = (
        longeval_ir_dataset
        and Path(longeval_ir_dataset).exists()
        and Path(longeval_ir_dataset).is_dir()
    )
    exists_in_irds = (
        longeval_ir_dataset in irds_registry and irds_registry[longeval_ir_dataset]
    )

    if exists_locally and exists_in_irds:
        raise ValueError(
            f"The passed {longeval_ir_dataset} is ambiguous, as it is a valid official ir_datasets id and a local directory."
        )

    if exists_locally:
        base = read_property_from_metadata(longeval_ir_dataset, "base")
        if base.startswith("longeval-sci"):
            LongEvalWebDataset(Path(longeval_ir_dataset))
        return LongEvalSciDataset(Path(longeval_ir_dataset))

    if exists_in_irds:
        return irds_registry[longeval_ir_dataset]

    raise ValueError(
        "I could not find a dataset with the id " + str(longeval_ir_dataset)
    )


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
