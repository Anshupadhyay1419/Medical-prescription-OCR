import re

from prescription_ocr.llm import prompts
from prescription_ocr.llm.client import generate


def rerank_candidates(candidates, timeout=15):
    """
    Return the candidate the LLM judges most plausible.

    Falls back to the first candidate — which is the greedy decode — whenever
    the LLM is unavailable or answers with something unusable.
    """
    seen, unique = set(), []
    for candidate in candidates:
        cleaned = candidate.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique.append(cleaned)

    if len(unique) <= 1:
        return unique[0] if unique else ""

    formatted = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(unique))

    try:
        result = generate(
            prompts.render("reranker", candidates=formatted),
            temperature=0.1,
            num_predict=10,
            timeout=timeout,
        )
        match = re.search(r'\d+', result)
        if match:
            index = int(match.group()) - 1
            if 0 <= index < len(unique):
                return unique[index]
    except Exception as e:
        print(f"  [Reranker] error: {e}")

    return unique[0]
