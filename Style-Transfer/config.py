# ==============================================================================
# Configuration constants for Prompt IDs used in style transfer and re-ranking.
# This file centralizes prompt selections to avoid naming conflicts across tasks.
# ==============================================================================

QUERY_PROMPT_NUMBER = 1
CORPUS_PROMPT_NUMBER = 1
RERANK_PROMPT_NUMBER = 5


def get_query_prompt() -> str:
    """
    Return the query prompt ID to use for style transfer.
    :return: Prompt number as a string.
    """
    return str(QUERY_PROMPT_NUMBER)


def get_corpus_prompt() -> str:
    """
    Return the corpus prompt ID to use for abstract generation.
    :return: Prompt number as a string.
    """
    return str(CORPUS_PROMPT_NUMBER)
