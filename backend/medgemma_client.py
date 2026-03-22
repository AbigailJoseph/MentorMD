"""Teaching brief client — previously MedGemma (Vertex AI), now backed by OpenAI."""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ENV_PATH = Path(__file__).resolve().with_name('.env')
load_dotenv(dotenv_path=ENV_PATH)

_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')


def query_medgemma(prompt: str, *, temperature: float = 0.2, max_tokens: int = 1024, top_p: float = 0.95) -> str:
    """Generate a structured teaching brief using OpenAI (drop-in replacement for the former MedGemma endpoint).

    Args:
        prompt: Prompt text containing the case narrative and Bayes summary.
        temperature: Sampling temperature for response diversity.
        max_tokens: Maximum output token budget.
        top_p: Nucleus sampling parameter.

    Returns:
        The model's teaching brief as a plain string.

    Raises:
        RuntimeError: If OPENAI_API_KEY is missing from the environment.
    """
    if not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError('Missing OPENAI_API_KEY in environment variables.')

    resp = _client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    return resp.choices[0].message.content.strip()
