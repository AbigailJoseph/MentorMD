"""Core tutoring pipeline that orchestrates parsing, inference, evaluation, and coaching."""

from typing import Dict, Any, List, Tuple
from pathlib import Path
from dataclasses import asdict

from dotenv import load_dotenv
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

from bayes.noisy_or_bayesnet import NoisyORBayesNet
from bayes.network_data import (
    PULMONARY_NETWORK_DATA,
    DISEASE_DISPLAY_NAMES,
    SYMPTOM_DISPLAY_NAMES,
)

from parsing.student_parser import StudentInputParser
from evaluation.diagnosis_evaluator import DiagnosisEvaluator
from evaluation.presentation_workflow import PresentationWorkflow
from agents.ai_attending import AIAttending
from pipeline.state import ConversationState
from medgemma_client import query_medgemma


CASE_NARRATIVE = """
Case: An 89-Year-Old Man with Progressive Dyspnea

An 89-year-old man was admitted to the hospital because of progressive dyspnea. The patient had been in his usual state of health, with diabetes mellitus, coronary artery disease, and complete heart block, until 6 months before admission, when shortness of breath developed. His exercise capacity gradually decreased from walking three blocks to climbing one flight of stairs.
Three days before admission, dyspnea increased, and he had difficulty walking two to three steps on a level plane. During the next 2 days, his children noted that he was somnolent and less interactive than usual, with decreased appetite and urine output and an unsteady gait. On the morning of admission, they called emergency medical services.
When the emergency medical services personnel arrived, the patient was sitting, with labored breathing. The blood pressure was 118/70 mm Hg, and the pulse 60 beats per minute; there were rales in both lung fields. The blood glucose level, by fingerstick measurement, was 230 mg per deciliter (12.8 mmol per liter), and an electrocardiogram (ECG) showed a paced rhythm with no ectopy.
During transport to this hospital by ambulance, oxygen (15 liters per minute) and furosemide were administered, with improvement. The patient did not have fevers, chills, night sweats, chest pain, or leg edema. A chronic, dry cough was unchanged. He had lost approximately 2.3 kg within the previous 2 months.
He had coronary artery disease (treated with a stent 10 years earlier), diabetes mellitus, hypertension, hyperlipidemia, cerebrovascular disease, osteoarthritis, hearing impairment, vertigo, and cataracts, and he had undergone a transurethral prostatectomy in the past. A diagnosis of complete heart block had been made 6 years earlier, when he presented with syncope; ECG at that time revealed an atrial rate of 120 beats per minute, with complete heart block and a ventricular escape rhythm of 30 to 33 beats per minute. A pacemaker was inserted.
Chest radiographs showed bilateral coarse reticular opacities, which were most pronounced at the lung bases. The patient lived alone and did his own shopping and cooking. He was a retired plumber who had been exposed to asbestos while working in a shipyard. He had smoked three packs of cigarettes per day for 50 years but had stopped 14 years earlier, and he drank alcohol occasionally; he did not use illicit drugs.
His mother had died at 49 years of age of unknown causes, his father had died of colon cancer, a brother had heart disease and asbestosis, and his adult children were well. Medications included metformin, rosiglitazone, acetylsalicylic acid, captopril, metoprolol, atorvastatin, furosemide, tolazamide, nifedipine, and cimetidine. He had no known allergies.
On examination in the emergency department, the temperature was 36.1°C, the blood pressure 105/43 mm Hg, and the pulse 81 beats per minute; the respirations were 28 per minute. The oxygen saturation was 78% while the patient was breathing ambient air, 91% with supplemental oxygen (6 liters) by nasal cannula, and 100% while he was breathing from a nonrebreather mask. The jugular venous pulse was visible at 14 cm above the right atrium. Crackles extended halfway up the lungs, with occasional expiratory wheezes; the remainder of the examination was normal.
Laboratory-test results are shown in Table 1. An ECG showed a pacemaker rate of 88 beats per minute. A chest radiograph showed bilateral patchy air-space opacities, indistinct pulmonary vessels, possible small pleural effusions, and a dual-lead pacemaker on the left chest wall. Additional furosemide and sodium polystyrene sulfonate were administered, and the patient was admitted to the medical service.
Later that day, while walking with assistance, a syncopal episode occurred. The systolic blood pressure was initially 60 mm Hg; 5 minutes later, his blood pressure was 140/68 mm Hg.
The next day, a chest radiograph revealed decreased air-space opacities, calcified pleural plaques on the right hemidiaphragm, and coarse reticular opacities, with increased cystic lucency, at both bases. Serial testing of cardiac biomarkers showed no evidence of myocardial infarction. Oxygen supplementation, captopril, atorvastatin, and metoprolol were continued; esomeprazole, insulin on a sliding scale for hyperglycemia, heparin, and low-dose acetylsalicylic acid were begun, with intravenous furosemide as needed for diuresis. Dyspnea on exertion persisted.
On the third day, oxygen saturation intermittently decreased to 77 to 85% while the patient was breathing 10 liters of oxygen with the use of a nonrebreather mask. A transthoracic echocardiogram showed mild concentric left ventricular hypertrophy and an ejection fraction of 54%. The right ventricle was markedly dilated and hypokinetic, with moderate-to-severe tricuspid-valve regurgitation, pulmonary arterial hypertension, and systolic and diastolic flattening of the interventricular septum, features that were consistent with both right ventricular volume and pressure overload.
Computed tomography (CT) of the chest revealed calcification of the pericardium and pleura, mild-to-moderate atherosclerosis in the aorta, and a dual-chamber pacemaker with leads in the innominate vein, right atrial appendage, and right ventricular septum. Paratracheal, hilar, and subcarinal lymph nodes ranged from 1.5 to 2.3 cm in diameter. Cystic changes and septal thickening were seen throughout both lungs, with bronchial-wall thickening, honeycomb changes, and traction bronchiectasis. There was diffuse asymmetric ground-glass opacification.
During the next 4 days, dyspnea and intermittent oxygen desaturation recurred with minimal activity, despite assistance by means of bilevel and continuous positive airway pressure. Ceftriaxone, azithromycin, levofloxacin, prednisone, morphine, bronchodilators, and additional furosemide were given. Laboratory-test results are shown in Table 1.
On the seventh day, the trachea was intubated; mechanical ventilation was begun, and the patient was transferred to the intensive care unit. Despite pressors, broad-spectrum antibiotics, morphine sulfate, stress-dose corticosteroids, insulin by intravenous infusion, intravenous fluids, and maximal ventilator support, his condition worsened, and on the eighth day, an asystolic cardiac arrest occurred. In consultation with the patient's family, cardiopulmonary resuscitation was not attempted, and the patient was pronounced dead. An autopsy was performed.
""".strip()


# ----------------------------
# BAYES SUMMARY
# ----------------------------

def build_bayes_summary(
    net: NoisyORBayesNet,
    evidence: Dict[str, bool],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Build a serializable evidence + ranked differential summary for downstream models."""

    ranked: List[Tuple[str, float]] = net.rank_diseases()

    return {
        "evidence": evidence,
        "top_differential": [
            {
                "diagnosis": d,
                "diagnosis_display": DISEASE_DISPLAY_NAMES.get(d, d),
                "probability": round(float(p), 4),
            }
            for d, p in ranked[:top_k]
        ],
        "evidence_display": [
            {
                "symptom": s,
                "symptom_display": SYMPTOM_DISPLAY_NAMES.get(s, s),
                "value": bool(v),
            }
            for s, v in evidence.items()
        ],
    }


# ----------------------------
# MEDGEMMA PROMPT
# ----------------------------

def build_medgemma_prompt(bayes_summary: Dict[str, Any]) -> str:
    """Create the grounded MedGemma prompt using case narrative and Bayes outputs."""
    return f"""
Create a concise, structured teaching brief using ONLY the provided information.
Do NOT invent patient facts. If something is missing, say it is missing.

CASE_NARRATIVE (fixed case background context for this scenario):
{CASE_NARRATIVE}

BAYES_NET_SUMMARY (ground truth probabilities):
{bayes_summary}

Return:
1) One-liner summary (1 sentence).
2) Top 3 diagnoses (use the Bayes list) with 1 supporting clue each.
3) 3 key missing discriminating questions (history/ROS).
4) 3 next best tests or imaging.
5) 2 short teaching pearls.

Keep it short and structured.
""".strip()


# ----------------------------
# PIPELINE
# ----------------------------

class ClinicalTutoringPipeline:
    """Stateful turn-based workflow for one tutoring session."""

    def __init__(self):
        """Initialize inference, parsing, evaluation, and coaching components."""
        self.bayes_net = NoisyORBayesNet(PULMONARY_NETWORK_DATA)
        self.state = ConversationState()

        self.parser = StudentInputParser()
        self.diagnosis_eval = DiagnosisEvaluator()
        self.attending = AIAttending()
        self.presentation_workflow = PresentationWorkflow()

    # STEP
    def step(self, student_input: str) -> str:
        """Process one student turn and return attending feedback."""

        # ---- FIRST TURN: RUN BAYES + MEDGEMMA + INITIAL EVALUATION ----
        if self.state.turn_number == 0:

            parsed = self.parser.parse(student_input)
            self.state.student_diagnoses = parsed.get("diagnoses", [])

            for s in parsed.get("present", []):
                if s not in self.state.symptoms_identified:
                    self.state.symptoms_identified.append(s)

            for s in parsed.get("absent", []):
                if s not in self.state.symptoms_absent:
                    self.state.symptoms_absent.append(s)

            evidence = {s: True for s in self.state.symptoms_identified}
            evidence.update({s: False for s in self.state.symptoms_absent})

            self.bayes_net.set_evidence(evidence)

            self.state.bayes_summary = build_bayes_summary(
                self.bayes_net, evidence
            )

            self.state.medgemma_packet = query_medgemma(
                build_medgemma_prompt(self.state.bayes_summary)
            )

            # 9-METRIC EVALUATION
            eval_result = self.presentation_workflow.evaluate_initial(
                student_input,
                case_narrative=CASE_NARRATIVE,
                bayes_summary=self.state.bayes_summary,
                medgemma_packet=self.state.medgemma_packet,
            )

            self.state.eval_packet = eval_result

        # ---- SUBSEQUENT TURNS: UPDATE EVALUATION ----
        else:
            parsed = self.parser.parse(student_input)
            if parsed.get("diagnoses"):
                self.state.student_diagnoses = parsed.get("diagnoses")

            eval_result = self.presentation_workflow.process_answer(
                student_input,
                case_narrative=CASE_NARRATIVE,
                bayes_summary=self.state.bayes_summary,
                medgemma_packet=self.state.medgemma_packet,
            )

            self.state.eval_packet = eval_result

            if eval_result.get("done") and eval_result.get(
                "turns_to_meet_all_metrics"
            ):
                self.state.turns_to_meet_all_metrics = eval_result[
                    "turns_to_meet_all_metrics"
                ]

        supported = any(
            self.diagnosis_eval.is_supported(self.bayes_net, dx)
            for dx in self.state.student_diagnoses
        )

        self.state.turn_number += 1

        return self.attending.respond(
            self.state,
            student_input=student_input,
            diagnosis_supported=supported,
        )

    # FINAL REPORT
    def final_evaluation(self) -> str:
        """Return a compact terminal-friendly summary of final performance."""
        summary = self.presentation_workflow.final_summary()

        return (
            f"Final Evaluation:\n"
            f"Metrics met: {summary.get('metrics_met')}\n"
            f"Total turns: {summary.get('turns')}\n"
            f"Turns to meet all metrics: {summary.get('turns_to_meet_all_metrics')}\n"
        )

    def to_snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot for durable session persistence."""
        return {
            "conversation_state": asdict(self.state),
            "presentation_workflow_state": self.presentation_workflow.export_state(),
            "attending_history": self.attending.export_history(),
        }

    @classmethod
    def from_snapshot(cls, snapshot: Dict[str, Any]) -> "ClinicalTutoringPipeline":
        """Rebuild a pipeline instance from a serialized snapshot."""
        pipeline = cls()

        state_data = snapshot.get("conversation_state", {}) or {}
        pipeline.state.turn_number = int(state_data.get("turn_number", 0))
        pipeline.state.symptoms_identified = list(state_data.get("symptoms_identified", []) or [])
        pipeline.state.symptoms_absent = list(state_data.get("symptoms_absent", []) or [])
        pipeline.state.student_diagnoses = list(state_data.get("student_diagnoses", []) or [])
        pipeline.state.bayes_summary = dict(state_data.get("bayes_summary", {}) or {})
        pipeline.state.medgemma_packet = str(state_data.get("medgemma_packet", ""))
        pipeline.state.eval_packet = dict(state_data.get("eval_packet", {}) or {})
        pipeline.state.turns_to_meet_all_metrics = state_data.get("turns_to_meet_all_metrics")
        pipeline.state.reasoning_progress = dict(state_data.get("reasoning_progress", {}) or {})

        evidence = {s: True for s in pipeline.state.symptoms_identified}
        evidence.update({s: False for s in pipeline.state.symptoms_absent})
        if evidence:
            pipeline.bayes_net.set_evidence(evidence)

        pipeline.presentation_workflow.import_state(
            snapshot.get("presentation_workflow_state", {}) or {}
        )
        pipeline.attending.import_history(snapshot.get("attending_history", []) or [])

        return pipeline
