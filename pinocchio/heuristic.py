import json
import os
from typing import List, Set, Tuple

import spacy
import torch
from nltk.corpus import stopwords
from torch.nn import CosineSimilarity

nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])
nlp_ner = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
stop_words = set(stopwords.words("english"))
stop_words.remove("only")
stop_words.update([",", ".", ")", "(", "-", ":", ""])

number_words = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "tens",
    "hundreds",
    "thousands",
}

cos = CosineSimilarity(dim=0)


def beam_search_token_scorer(
    predicted_token: str,
    previous_token: str,
    previous_previous_token: str,
    entropy: float,
    previous_entropy: float,
    previous_previous_entropy: float,
    top_predicted_tokens: List[str],
    attribution: List[Tuple[float, str]],
    previous_context: str,
    unmasker,
    previous_dead_end_tokens: Set[str],
    abstractive: bool,
    bart_model,
    tokenizer,
    without_confidence: bool = False,
    without_attribution: bool = False,
    without_common: bool = False,
):
    if predicted_token.isspace() or predicted_token == "</s>":
        return 1

    if predicted_token in previous_dead_end_tokens:
        return 0

    index = None
    next_attribution_strings = None
    next_attribution_matches = None

    original_predicted_token = predicted_token

    multi_token = False
    if (
        not predicted_token == "</s>"
        and (
            (not predicted_token.strip().lower() in stop_words)
            and predicted_token.startswith(" ")
        )
        and (
            (not previous_token.strip().lower() in stop_words)
            and previous_token.startswith(" ")
        )
        and previous_token.startswith(" ")
        and not predicted_token.startswith(" ")
    ):
        new_predicted_token = previous_token + predicted_token
        if True or new_predicted_token.isupper():
            multi_token = True
            entropies = [e for e in [entropy, previous_entropy] if e != -1]
            entropy = sum(entropies) / len(entropies)
            predicted_token = new_predicted_token

    predicted_token_lemma = " ".join(
        [t.lemma_.lower() for t in nlp(predicted_token.strip())]
    )
    attribution_scores = [e[0] for e in attribution]
    not_start_attribution_scores = [
        e[0] for e in attribution if not e[1].startswith("<s>")
    ]
    attribution_sum = sum(attribution_scores)
    attribution_average = attribution_sum / len(attribution_scores)
    attribution_strings = [
        " ".join([t.lemma_.lower() for t in nlp(e[1])]) for e in attribution
    ]
    attribution_starts = [
        1 if "< s >" in string else 0 for string in attribution_strings
    ]
    attribution_starts_sum = sum(attribution_starts)
    attribution_matches = [
        1
        if predicted_token_lemma.strip().lower() in attribution_string.strip().lower()
        else 0
        for attribution_string in attribution_strings
    ]

    matching_score = sum(
        [attribution_scores[i] for i in range(5) if attribution_matches[i] == 1]
    )
    not_start_matching_score = sum(
        [
            attribution_scores[i]
            for i in range(5)
            if attribution_matches[i] == 1 and not attribution[i][1].startswith("<s>")
        ]
    )
    next_token_matches = [
        1
        if any(
            (token.isspace() or (nlp(token.strip())[0].lemma_.lower() in att))
            for att in attribution_strings
        )
        else 0
        for token in top_predicted_tokens
        if token != " "
    ]
    next_token_stopwords = [
        1 if token.lower().strip() in stop_words else 0
        for token in top_predicted_tokens
    ]

    if abstractive:
        predicted_token_embedding = bart_model.shared(
            torch.LongTensor([tokenizer.encode(predicted_token)[1]]).to(
                bart_model.device
            )
        ).squeeze()

        attribution_token_ids = [
            [
                tokenizer.encode(word.text)[1:-1]
                for word in nlp.tokenizer(a[1])
                if not word.text.strip().lower() in stop_words
            ]
            for a in attribution
        ]

        # pool across the embeddings
        attribution_max_pooled_embeddings = [
            [
                bart_model.shared(
                    torch.LongTensor(token_ids).to(bart_model.device)
                ).max(dim=0)[0]
                for token_ids in attribution_token_ids[i]
            ]
            for i in range(5)
        ]

        max_cos_sim = max(
            [
                max(
                    [
                        cos(predicted_token_embedding, attribution_single_word).item()
                        for attribution_single_word in attribution_single_sequence
                    ]
                )
                if len(attribution_single_sequence) > 0
                else 0
                for attribution_single_sequence in attribution_max_pooled_embeddings
            ]
        )

    if (
        (
            (entropy < (0.75 if not abstractive else 1.0))
            and (predicted_token in top_predicted_tokens)
            and (top_predicted_tokens.index(predicted_token) < 2)
            and (not without_confidence)
        )
        or (
            (
                sum(attribution_matches) >= 3
                or attribution_matches[0] == 1
                or matching_score > (sum(attribution_scores) / 3)
                # or (abstractive and not_start_matching_score > (sum(not_start_attribution_scores) / 2))
                or (
                    (not multi_token)
                    and (
                        not abstractive
                        or (
                            predicted_token in top_predicted_tokens
                            and top_predicted_tokens.index(predicted_token) < 2
                        )
                    )
                    and (
                        sum(next_token_matches)
                        >= ((10 - (sum(next_token_stopwords))) / 2)
                    )
                )
            )
            and (not without_attribution)
        )
        or (
            not abstractive
            and (
                (
                    original_predicted_token.strip().lower() in stop_words
                    and original_predicted_token.startswith(" ")
                )
            )
            and (not without_common)
        )
        or (
            abstractive
            and (max_cos_sim >= 0.15)
            and not predicted_token.strip()[0].isupper()
            and (
                predicted_token.strip().lower() not in number_words
                and not predicted_token.strip().lower().isnumeric()
            )
            and (not without_common)
        )
    ):
        return 1
    else:
        previous_context_with_mask = previous_context + "<mask>"
        unmasked = unmasker(previous_context_with_mask.lower())

        if (
            (not multi_token)
            and any(
                output["sequence"][4:-4].strip()
                == (previous_context + original_predicted_token)
                and output["score"] > 0.01
                for output in unmasked
            )
            and (not without_common)
        ):
            return 1
    return 0
