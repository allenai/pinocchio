import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import fire
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from pinocchio.text_utils import generic_text_predictions
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline


def main(
    output_file: str,
    gpu_id: int,
    split: Optional[str] = "test",
    seed: Optional[int] = 42,
    resume_index: Optional[int] = 0,
    end_index: Optional[int] = None,
):
    random.seed(seed)
    np.random.seed(seed)

    if end_index is None:
        dataset = load_dataset("xsum", split=f"{split}[{resume_index}:]")
    else:
        dataset = load_dataset("xsum", split=f"{split}[{resume_index}:{end_index}]")
    print(len(dataset))

    print("loading bart...")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-xsum")
    bart = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-large-xsum", output_attentions=True
    )
    bart.to(f"cuda:{gpu_id}")
    print("bart loaded.")

    print("Loading unmasker...")
    unmasker = pipeline("fill-mask", model="roberta-base", device=gpu_id)
    print("Loaded unmasker.")

    for i, input_item in enumerate(tqdm(dataset, desc="Predicting...")):
        output_dict_example = {}
        input_text = input_item["document"]
        doc_id = input_item["id"]
        gold = input_item["summary"]

        if len(input_text) == 0 or len(gold) == 0:
            continue

        (
            summary,
            predictions,
            full_entropy,
            best_score,
            num_resets,
        ) = generic_text_predictions(
            bart,
            tokenizer,
            unmasker,
            [input_text],
            gpu_id,
            use_filter=True,  # Setting use_filter to true to use the pinocchio algorithm
            abstractive=True,
            print_beam_info=False,
            seed=1111,
        )

        if summary is None:
            continue

    output_dict_example["id"] = doc_id
    output_dict_example["predicted"] = summary
    output_dict_example["gold"] = gold

    output_dict_example["num_resets"] = num_resets
    output_dict_example["best_scores"] = best_score
    output_dict_example["full_entropy"] = full_entropy
    output_dict_example["full_output"] = predictions

    with open(output_file, "a") as _jsonl_file:
        _jsonl_file.write(json.dumps(output_dict_example))
        _jsonl_file.write("\n")


if __name__ == "__main__":
    fire.Fire(main)
