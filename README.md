# MentorMD: An AI Attending Physician

MentorMD is an AI attending physician that helps medical students practice clinical reasoning by analyzing case presentations, evaluating diagnostic thinking, and guiding learners with targeted Socratic feedback.

### Team Members
- Katie Xiao: Full-Stack Development (Backend, Firebase, Frontend)
- Parvi Chadha: AI Agent Design & Prompt Engineering
- Abigail Joseph: Frontend Development & System Integration
- Caleb Kim: Bayesian Network & Backend Prompt Engineering

---

## What The System Does

- Accepts a student's initial case presentation.
- Extracts likely symptoms/diagnoses from free text.
- Uses a Noisy-OR Bayes network to ground the differential.
- Generates a MedGemma knowledge packet from grounded context.
- Evaluates presentation quality across 9 clinical communication/reasoning metrics.
- Produces attending-style coaching plus one targeted question per turn.
- Stores case outcomes, transcript, and profile stats for longitudinal learning.

---

## Quick Start (Local Development)

### 1. Configure environment variables

### Environment Variables

Backend `backend/.env`
```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=your_openai_model
MEDGEMMA_PROJECT_ID=your_medgemma_project_id
MEDGEMMA_ENDPOINT_ID=your_medgemma_endpoint_id
FIREBASE_SERVICE_ACCOUNT_KEY=your_firebase_service_account_key
ALLOWED_ORIGINS=your_allowed_origins 
```

Frontend (`frontend/.env`)
```env
VITE_FIREBASE_API_KEY=your_firebase_web_api_key
VITE_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=your-project-id
VITE_FIREBASE_STORAGE_BUCKET=your-project.firebasestorage.app
VITE_FIREBASE_MESSAGING_SENDER_ID=1234567890
VITE_FIREBASE_APP_ID=1:1234567890:web:abcdef1234567890
VITE_FIREBASE_MEASUREMENT_ID=G-XXXXXXXXXX
VITE_API_BASE_URL=http://localhost:8000
```

### 2. Run backend 

```bash
cd backend
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
```

### 3. Run frontend

```bash
cd frontend
npm install
npm run dev
```

---


## Repository Structure 

```text
MentorMD/
|-- backend/
|   |-- main.py                               # CLI entrypoint for local tutoring session
|   |-- server.py                             # FastAPI API 
|   |-- medgemma_client.py                    # Backend MedGemma Vertex endpoint wrapper
|   |-- agents/
|   |   `-- ai_attending.py                   # Attending-style coaching response generator
|   |-- parsing/
|   |   `-- student_parser.py                 # Extracts symptoms/differential from student text
|   |-- pipeline/
|   |   |-- pipeline.py                       # Main orchestration: parse -> infer -> evaluate -> respond
|   |   `-- state.py                          # Conversation state dataclass across turns
|   |-- evaluation/
|   |   |-- presentation_workflow.py          # 9-metric rubric evaluation + Socratic question loop
|   |   `-- diagnosis_evaluator.py            # Checks diagnosis support against Bayes outputs
|   `-- bayes/
|       |-- noisy_or_bayesnet.py              # Noisy-OR Bayesian inference engine
|       `-- network_data.py            
`-- frontend/
    `-- src/         
```

---

## System Diagram

![MentorMD System Diagram](docs/system-diagram.png)

## High-Level Flow

1. User signs in on the frontend via Firebase Auth.
2. Frontend sends authenticated requests to backend session endpoints.
3. Backend parses student input into structured symptoms and diagnoses.
4. Bayes engine computes grounded differential probabilities.
5. Backend builds a MedGemma prompt packet using case context and Bayes summary.
6. Evaluation workflow grades the presentation across 9 metrics and identifies gaps.
7. AI attending returns concise coaching and one targeted Socratic question.
8. Final case performance and transcript are persisted in Firestore and shown in profile analytics.

---

## Technical Details 

### Tech Stack

Frontend:
- React
- TypeScript
- Vite
- Firebase Authentication
- Cloud Firestore

Backend:
- Python
- FastAPI
- OpenAI API
- Google Vertex AI endpoint (MedGemma)
- Firebase Admin SDK
- Custom Noisy-OR Bayesian network

### Evaluation Rubric 

MentorMD evaluates case presentations across:
1. Focused relevant information selection
2. Clear working diagnosis statement
3. Logical organization and reasoning
4. Prioritized differential diagnosis
5. Conciseness and efficient delivery
6. Rational diagnostic workup prioritization
7. Management plan and disposition prioritization
8. Hypothesis-driven inquiry
9. Ability to synthesize findings
