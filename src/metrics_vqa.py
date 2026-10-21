"""
VQAv2 accuracy scoring.

Reference: Antol et al., "VQA: Visual Question Answering", ICCV 2015,
also Goyal et al., "Making the V in VQA Matter", CVPR 2017.

The standard VQA accuracy metric is:

    Per-question score:
        score(answer) = min( |{h: h_answer == answer}| / 3, 1.0 )

    where h_answer ranges over the 10 human-annotated answers for the
    question. So if 3+ humans gave that answer you get 1.0; 2 humans -> 2/3;
    1 human -> 1/3; 0 humans -> 0.

    Robust averaging:
        Each predicted answer is compared against each of the 10 GT answers,
        but each comparison samples 9 of the 10 humans. The final score
        averages over all 10 (size-9) leave-one-out samples. This is the
        "accuracy" used by the official evaluation server.

We implement both:
  - vqa_accuracy_simple()   — the simpler min(matches/3, 1.0) version,
                              fine for our purposes (correlates >0.99 with
                              the leave-one-out version)
  - vqa_accuracy_loo()      — the official leave-one-out version, for
                              direct comparison to other VQAv2 results

Answer normalization (REQUIRED before comparison):
  - lowercase
  - strip punctuation (.,!?:;)
  - strip leading articles (a, an, the)
  - collapse internal whitespace
  - some standard number-word -> digit conversions (one->1, two->2, etc.)

The official VQA eval code does additional contraction/period handling;
we replicate the most important of these transforms.
"""

import re
from collections import defaultdict


# Article words to strip
_ARTICLES = {"a", "an", "the"}

# Number-word to digit (very common in VQA answers)
_NUMBER_WORDS = {
    "none": "0", "zero": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

# Common contractions in VQA answers (subset of the official eval code's list)
_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't",
    "couldve": "could've", "couldnt": "couldn't", "didnt": "didn't",
    "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hasnt": "hasn't", "havent": "haven't", "isnt": "isn't",
    "shouldnt": "shouldn't", "wasnt": "wasn't", "werent": "weren't",
    "wont": "won't", "wouldnt": "wouldn't",
    "im": "i'm", "ive": "i've", "thats": "that's", "theres": "there's",
    "theyre": "they're", "whats": "what's", "wheres": "where's",
    "youre": "you're", "youve": "you've",
}

# Punctuation pattern (everything except apostrophes and hyphens)
_PUNCT_PATTERN = re.compile(r"[.,!?;:\"\[\]{}()]")


def normalize_answer(ans):
    """
    Apply the canonical VQA answer-normalization sequence.

    Args:
        ans: a string answer (predicted or GT)
    Returns:
        normalized string
    """
    if ans is None:
        return ""

    s = str(ans).lower().strip()

    # Strip punctuation, normalize whitespace
    s = _PUNCT_PATTERN.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Tokenize
    tokens = s.split()

    # Strip leading articles
    while tokens and tokens[0] in _ARTICLES:
        tokens.pop(0)

    # Replace number words with digits
    tokens = [_NUMBER_WORDS.get(t, t) for t in tokens]

    # Apply contraction normalization
    tokens = [_CONTRACTIONS.get(t, t) for t in tokens]

    return " ".join(tokens)


def vqa_accuracy_simple(pred, gt_answers):
    """
    Per-question VQA score using the simple min(matches/3, 1.0) rule.

    Args:
        pred: predicted answer string
        gt_answers: list of 10 GT answer strings (one per human annotator)
    Returns:
        float in [0, 1]
    """
    pn = normalize_answer(pred)
    matches = sum(1 for g in gt_answers if normalize_answer(g) == pn)
    return min(matches / 3.0, 1.0)


def vqa_accuracy_loo(pred, gt_answers):
    """
    Per-question VQA score using the official leave-one-out averaging.

    For each i in 0..9: count matches in the 9-element list excluding GT[i],
    score = min(matches/3, 1.0). Final score = mean over all i.

    Args:
        pred: predicted answer string
        gt_answers: list of 10 GT answer strings
    Returns:
        float in [0, 1]
    """
    pn = normalize_answer(pred)
    norm_gts = [normalize_answer(g) for g in gt_answers]

    scores = []
    for i in range(len(norm_gts)):
        remaining = norm_gts[:i] + norm_gts[i+1:]
        matches = sum(1 for g in remaining if g == pn)
        scores.append(min(matches / 3.0, 1.0))
    return sum(scores) / len(scores)


def aggregate_vqa_scores(per_question_results, scorer="loo"):
    """
    Compute mean VQA accuracy overall and broken down by answer type.

    Args:
        per_question_results: list of dicts, each with keys:
            'pred': predicted answer string
            'gt_answers': list of 10 GT strings
            'answer_type': one of {'yes/no', 'number', 'other'}
        scorer: 'loo' (official) or 'simple'
    Returns:
        dict with 'overall', 'yes/no', 'number', 'other' float scores
        and 'n_*' counts.
    """
    fn = vqa_accuracy_loo if scorer == "loo" else vqa_accuracy_simple

    per_type_sum = defaultdict(float)
    per_type_count = defaultdict(int)
    total_sum = 0.0

    for r in per_question_results:
        s = fn(r["pred"], r["gt_answers"])
        per_type_sum[r["answer_type"]] += s
        per_type_count[r["answer_type"]] += 1
        total_sum += s

    n_total = sum(per_type_count.values())
    overall = total_sum / max(n_total, 1)

    out = {"overall": overall * 100.0, "n_total": n_total}
    for t in ["yes/no", "number", "other"]:
        n = per_type_count.get(t, 0)
        out[t] = (per_type_sum.get(t, 0.0) / n * 100.0) if n > 0 else 0.0
        out[f"n_{t}"] = n
    return out
