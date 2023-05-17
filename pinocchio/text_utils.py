import random
import re
from typing import List

import numpy as np
import spacy
import textacy
import torch
from fuzzysearch import find_near_matches

nlp_sci_sm = spacy.load("en_core_sci_sm")


def generic_text_predictions(
    model,
    tokenizer,
    unmasker,
    batch,
    device,
    use_filter,
    abstractive,
    print_beam_info,
    seed=1111,
    without_confidence=False,
    without_attribution=False,
    without_common=False,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != "-1":
        torch.cuda.manual_seed_all(seed)

    dct = tokenizer.batch_encode_plus(batch, max_length=1024, return_tensors="pt", pad_to_max_length=True,)
    if device != "-1":
        input_ids = dct["input_ids"].to(device)
        attention_mask = dct["attention_mask"].to(device)
    else:
        input_ids = dct["input_ids"]
        attention_mask = dct["attention_mask"]

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        min_length=5,
        max_length=500,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=2.0,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        num_beams=6,
        tokenizer=tokenizer,
        extractive_adjustment=0.0,
        do_sample=False,
        return_score=False,
        unmasker=unmasker,
        use_filter=use_filter,
        abstractive=abstractive,
        print_beam_info=print_beam_info,
        without_confidence=without_confidence,
        without_attribution=without_attribution,
        without_common=without_common,
    )

    if output is None:
        return [None] * 5

    preds = [tokenizer.decode(g[2:], skip_special_tokens=False, clean_up_tokenization_spaces=True) for g in output[0]]

    return (
        preds,
        [
            [
                (
                    tokenizer.decode(pred_word.item(), skip_special_tokens=False, clean_up_tokenization_spaces=True,),
                    entropy_word.item(),
                    top_k_word,
                    attribution_word,
                )
                for pred_word, entropy_word, top_k_word, attribution_word in zip(
                    pred[2:], entropy[2:], top_k[2:], attribution[2:]
                )
            ]
            for pred, entropy, top_k, attribution in zip(output[0].detach().cpu(), output[1], output[2], output[3])
        ],
        [sum(entropy[2:]).item() / len(entropy[2:]) for entropy in output[1]],
        [output[4]],
        output[5],
    )