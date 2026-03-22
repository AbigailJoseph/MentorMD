"""Conversational AI attending that coaches students turn-by-turn."""

import os
from pathlib import Path
from typing import Dict, Any, List

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)


SYSTEM_PROMPT = """You are the AI Attending Physician (AI-AP) coaching a medical student.

Hard rules:
1) Base all feedback strictly on the clinical context, knowledge, and evaluation data provided to you. Never invent patient facts.
2) Never reference internal data sources, tools, or system components by name (e.g. do not say "Bayes net", "MedGemma", "knowledge packet", "evaluation packet", or any similar terms).
3) Speak naturally as an attending physician — your reasoning should feel clinical, not computational.
4) Be constructive, specific, and brief.

Output format every time (strict):
- 1-2 sentences: Coaching feedback grounded in the clinical information provided.
- End with EXACTLY ONE question to probe the student's reasoning (prefer questions from the evaluation data if available).
"""


class AIAttending:
    """Wrapper around OpenAI chat completions for attending-style feedback."""

    def __init__(self, model: str = None):
        if OpenAI is None:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        # Stores interleaved user/assistant turns for conversation continuity
        self._history: List[Dict[str, str]] = []

    def initial_message(self, bayes_summary: Dict[str, Any], medgemma_packet: str) -> str:
        """Generate a first-turn kickoff message before student input."""
        context = self._make_context(bayes_summary, medgemma_packet, student_state={"turn_number": 0})
        return self._chat(context, user_message="(Start the session.)")

    def respond(self, state, student_input: str, diagnosis_supported: bool) -> str:
        """Generate attending feedback for a student's current turn."""
        student_state = {
            "turn_number": state.turn_number,
            "student_diagnoses": state.student_diagnoses,
            "diagnosis_supported": diagnosis_supported,
            "symptoms_identified": state.symptoms_identified,
        }
        context = self._make_context(state.bayes_summary, state.medgemma_packet, student_state, eval_packet=state.eval_packet)
        return self._chat(context, user_message=student_input)

    def _make_context(self, bayes_summary, medgemma_packet, student_state, eval_packet=None) -> str:
        """Build the developer-context block passed to the LLM each turn."""
        return f"""BAYES_NET_SUMMARY:
{bayes_summary}

MEDGEMMA_KNOWLEDGE_PACKET:
{medgemma_packet}

EVALUATION_SUMMARY:
{eval_packet.get("evaluation", {})}

EVALUATION_QUESTIONS:
{eval_packet.get("questions", [])}

STUDENT_STATE:
{student_state}
"""

    def _chat(self, context: str, user_message: str) -> str:
        """Call OpenAI chat completions and persist conversation history."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            # Developer block is rebuilt each turn with the latest Bayes/eval state
            {"role": "developer", "content": context},
            # Prior conversation turns give the model memory of what was already discussed
            *self._history,
            {"role": "user", "content": user_message},
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
        )
        reply = resp.choices[0].message.content.strip()
        # Append this turn to history so it's available on the next call
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": reply})
        return reply

    def export_history(self) -> List[Dict[str, str]]:
        """Return a JSON-serializable copy of conversation history."""
        return [dict(item) for item in self._history]

    def import_history(self, history: List[Dict[str, str]]) -> None:
        """Restore conversation history from serialized data."""
        self._history = [dict(item) for item in history if isinstance(item, dict)]
