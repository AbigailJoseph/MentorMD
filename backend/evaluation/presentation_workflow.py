"""Presentation evaluation workflow for iterative coaching over 9 rubric metrics."""

import json
import os
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

from openai import OpenAI


METRICS_RUBRIC = """
EVALUATION METRICS FOR MEDICAL CASE PRESENTATIONS:

GRADING PHILOSOPHY — be fair, not perfect. Students do not need to use ideal phrasing
or cover every sub-point. "Met" means the student demonstrated reasonable competency in
the area. "Partial" means they addressed it but missed something meaningful. "Missing"
means the concept was not addressed at all. Do not penalize for brevity if the substance
is present.

    1. FOCUSED, RELEVANT INFORMATION SELECTION (MOST IMPORTANT)
       - MET if: student's presentation is clearly filtered toward their working diagnosis
         (doesn't have to be perfect, just demonstrates active selection)
       - PARTIAL if: student includes mostly relevant details but also some unnecessary ones
       - MISSING if: student recites all available facts with no apparent filtering

    2. CLEAR STATEMENT OF WORKING DIAGNOSIS
       - MET if: student explicitly names a working diagnosis and briefly justifies it
       - PARTIAL if: student names a diagnosis but provides no supporting rationale,
         OR implies it through workup without stating it
       - MISSING if: no diagnosis or diagnostic direction is mentioned

    3. LOGICAL ORGANIZATION + CLINICAL REASONING
       - MET if: student connects findings to their reasoning (even a single "because" or
         "given X, I think Y" counts)
       - PARTIAL if: student lists findings and a diagnosis but does not link them
       - MISSING if: presentation is a disorganized data dump with no reasoning shown

    4. INCLUSION OF PRIORITIZED DIFFERENTIAL DIAGNOSIS
       - MET if: student names at least one alternative diagnosis with some rationale
       - PARTIAL if: student lists alternatives but with no ordering or reasoning
       - MISSING if: no alternative diagnoses are mentioned

    5. CONCISENESS + EFFICIENT DELIVERY
       - MET if: presentation is reasonably concise and structured
       - PARTIAL if: noticeably verbose or repetitive but still coherent
       - MISSING if: severely disorganized or filled with irrelevant content

    6. PRIORITIZED, RATIONAL DIAGNOSTIC WORKUP PLAN
       - MET if: student names relevant tests and explains why they matter or what order
       - PARTIAL if: student lists tests but gives no prioritization or rationale
       - MISSING if: no workup plan is mentioned

    7. PRIORITIZED MANAGEMENT PLAN AND DISPOSITION
       - MET if: student outlines a reasonable management approach with some ordering
       - PARTIAL if: student mentions treatment/disposition but it is incomplete or unordered
       - MISSING if: no management or disposition is mentioned

    8. EVIDENCE OF HYPOTHESIS-DRIVEN INQUIRY
       - MET if: student's plan or questions clearly flow from their stated hypothesis
       - PARTIAL if: student has a hypothesis but the workup/plan doesn't obviously follow from it
       - MISSING if: no discernible hypothesis is driving the student's approach

    9. ABILITY TO SYNTHESIZE (NOT JUST REPORT)
       - MET if: student offers a summary statement or interpretation that goes beyond
         listing facts (e.g. "this presentation is most consistent with X because...")
       - PARTIAL if: student states a conclusion but doesn't connect it to the data
       - MISSING if: student only lists findings with no interpretation
""".strip()


@dataclass
class EvalState:
    """Mutable workflow state tracked across follow-up interactions."""
    interaction_count: int = 0
    max_interactions: int = 15
    initial_presentation: str = ""
    # metric_id -> {name, status, confidence}
    metrics_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    initial_evaluation: Optional[Dict[str, Any]] = None
    all_metrics_met_turn: Optional[int] = None  # turn index when achieved


class PresentationWorkflow:
    """
    OpenAI-based version of hackathon workflow:
    - Evaluate initial presentation against 9 metrics
    - Generate probing questions for missing/partial metrics
    - Track how many turns until all metrics are met
    """

    def __init__(self, model: Optional[str] = None):
        """Initialize OpenAI client and in-memory state."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.state = EvalState()

    def reset(self):
        """Clear workflow state for a new presentation session."""
        self.state = EvalState()

    def export_state(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of workflow state."""
        return asdict(self.state)

    def import_state(self, state_data: Dict[str, Any]) -> None:
        """Restore workflow state from a serialized snapshot."""
        self.state = EvalState(
            interaction_count=int(state_data.get("interaction_count", 0)),
            max_interactions=int(state_data.get("max_interactions", 15)),
            initial_presentation=state_data.get("initial_presentation", ""),
            metrics_status=state_data.get("metrics_status", {}) or {},
            conversation_history=state_data.get("conversation_history", []) or [],
            initial_evaluation=state_data.get("initial_evaluation"),
            all_metrics_met_turn=state_data.get("all_metrics_met_turn"),
        )

    def evaluate_initial(
        self,
        student_presentation: str,
        *,
        case_narrative: str,
        bayes_summary: Dict[str, Any],
        medgemma_packet: str,
    ) -> Dict[str, Any]:
        """Evaluate the initial student presentation and generate first questions."""
        self.reset()
        self.state.interaction_count = 0
        self.state.initial_presentation = student_presentation

        evaluation = self._evaluate_presentation(
            student_presentation,
            case_narrative=case_narrative,
            bayes_summary=bayes_summary,
            medgemma_packet=medgemma_packet,
        )
        self.state.initial_evaluation = evaluation

        self._hydrate_metrics_status(evaluation)

        gaps = [e for e in evaluation["evaluations"] if e["status"] in ["missing", "partial", "misconception"]]
        questions = self._generate_questions(
            missing_metrics=gaps[:3],
            case_narrative=case_narrative,
            bayes_summary=bayes_summary,
            medgemma_packet=medgemma_packet,
            conversation_history=[],
        )

        # store questions so you can evaluate answers later if you want
        for q in questions:
            self.state.conversation_history.append({"question": q, "answer": None})

        return {
            "evaluation": evaluation,
            "questions": questions,
            "metrics_status": self.state.metrics_status,
        }

    def process_answer(
        self,
        student_answer: str,
        *,
        case_narrative: str,
        bayes_summary: Dict[str, Any],
        medgemma_packet: str,
    ) -> Dict[str, Any]:
        """Process a student follow-up response and continue rubric tracking."""
        self.state.interaction_count += 1

        # attach answer to all currently unanswered questions (student replies to the batch)
        for turn in self.state.conversation_history:
            if turn["answer"] is None:
                turn["answer"] = student_answer

        # Re-evaluate the *current* presentation state as:
        # original presentation + conversation so far (simple and robust)
        stitched = self._stitch_presentation()

        evaluation = self._evaluate_presentation(
            stitched,
            case_narrative=case_narrative,
            bayes_summary=bayes_summary,
            medgemma_packet=medgemma_packet,
        )
        self._hydrate_metrics_status(evaluation)

        remaining = [m for m in evaluation["evaluations"] if m["status"] in ["missing", "partial", "misconception"]]

        if not remaining and self.state.all_metrics_met_turn is None:
            self.state.all_metrics_met_turn = self.state.interaction_count

        if not remaining:
            return {
                "done": True,
                "evaluation": evaluation,
                "metrics_status": self.state.metrics_status,
                "turns_to_meet_all_metrics": self.state.all_metrics_met_turn,
            }

        if self.state.interaction_count >= self.state.max_interactions:
            return {
                "done": True,
                "timeout": True,
                "evaluation": evaluation,
                "metrics_status": self.state.metrics_status,
                "turns_to_meet_all_metrics": None,
            }

        questions = self._generate_questions(
            missing_metrics=remaining[:3],
            case_narrative=case_narrative,
            bayes_summary=bayes_summary,
            medgemma_packet=medgemma_packet,
            conversation_history=self.state.conversation_history[-4:],
        )
        for q in questions:
            self.state.conversation_history.append({"question": q, "answer": None})

        return {
            "done": False,
            "evaluation": evaluation,
            "questions": questions,
            "metrics_status": self.state.metrics_status,
        }

    def final_summary(self) -> Dict[str, Any]:
        """Return aggregate evaluation progress and conversation history."""
        met = sum(1 for m in self.state.metrics_status.values() if m["status"] == "met")
        total = len(self.state.metrics_status) if self.state.metrics_status else 9
        return {
            "metrics_met": f"{met}/{total}",
            "turns": self.state.interaction_count,
            "turns_to_meet_all_metrics": self.state.all_metrics_met_turn,
            "metrics_status": self.state.metrics_status,
            "conversation_history": self.state.conversation_history,
        }

    # ---------------- internal helpers ----------------

    def _stitch_presentation(self) -> str:
        """Combine initial presentation with answered Q/A turns for reevaluation."""
        parts = [self.state.initial_presentation]
        for t in self.state.conversation_history:
            if t.get("question") and t.get("answer"):
                parts.append(f"Q: {t['question']}\nA: {t['answer']}")
        return "\n\n".join(parts).strip()

    def _hydrate_metrics_status(self, evaluation: Dict[str, Any]) -> None:
        """Update metric snapshot from the latest evaluator payload."""
        # Build/update metrics_status snapshot
        ms: Dict[str, Dict[str, Any]] = {}
        for e in evaluation.get("evaluations", []):
            ms[e["metric_id"]] = {
                "name": e["metric_name"],
                "status": e["status"],
                "confidence": e["confidence"],
            }
        # If evaluator ever fails to return 9, keep old values
        if len(ms) >= 6:
            self.state.metrics_status = ms

    def _evaluate_presentation(
        self,
        student_text: str,
        *,
        case_narrative: str,
        bayes_summary: Dict[str, Any],
        medgemma_packet: str,
    ) -> Dict[str, Any]:
        """Run rubric grading with OpenAI and parse the JSON response."""
        prompt = f"""
You are an expert medical attending physician evaluating a student's case presentation.

{METRICS_RUBRIC}

GROUNDING CONTEXT (do not invent facts):
CASE_NARRATIVE:
{case_narrative}

BAYES_NET_SUMMARY:
{json.dumps(bayes_summary, indent=2)}

MEDGEMMA_KNOWLEDGE_PACKET:
{medgemma_packet}

STUDENT TEXT TO EVALUATE:
---
{student_text}
---

Evaluate against ALL 9 metrics using the GRADING PHILOSOPHY above.
Award "met" when the student demonstrates reasonable competency in the area — they do not
need to be exhaustive or perfectly worded. Use "partial" when they addressed the concept
but missed something meaningful. Use "missing" only when the concept is absent entirely.

For each metric return:
- status: "met", "partial", "missing", or "misconception"
- confidence: 0.0 to 1.0
- evidence: short quote/observation
- gaps: short missing items (only if not "met")

CRITICAL: Return ONLY valid JSON.

Return format:
{{
  "evaluations": [
    {{
      "metric_id": "1",
      "metric_name": "Focused, Relevant Information Selection",
      "status": "met|partial|missing|misconception",
      "confidence": 0.0,
      "evidence": "...",
      "gaps": "..."
    }}
  ],
  "overall_assessment": "...",
  "priority_gaps": ["..."]
}}
""".strip()

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(resp.choices[0].message.content)

    def _generate_questions(
        self,
        *,
        missing_metrics: List[Dict[str, Any]],
        case_narrative: str,
        bayes_summary: Dict[str, Any],
        medgemma_packet: str,
        conversation_history: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate targeted Socratic follow-up questions for current gaps."""
        history_text = "\n".join(
            [f"Q: {t['question']}\nA: {t.get('answer','')}" for t in conversation_history if t.get("question")]
        ).strip() or "None."

        prompt = f"""
You are a medical attending using the Socratic method.

Missing/Partial/Misconception metrics to address (top priority first):
{json.dumps(missing_metrics, indent=2)}

GROUNDING CONTEXT:
CASE_NARRATIVE:
{case_narrative}

BAYES_NET_SUMMARY:
{json.dumps(bayes_summary, indent=2)}

MEDGEMMA_KNOWLEDGE_PACKET:
{medgemma_packet}

Recent conversation:
{history_text}

Generate 1-2 targeted, open-ended questions that push the student to address the gaps.
Avoid yes/no. Keep each question <= 25 words.

Return ONLY JSON:
{{ "questions": ["...", "..."] }}
""".strip()

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        data = json.loads(resp.choices[0].message.content)
        return [q for q in data.get("questions", []) if isinstance(q, str)]
    


    
