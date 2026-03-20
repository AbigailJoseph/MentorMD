# MentorMD: An AI Attending Physician

MentorMD is an AI attending physician that helps medical students practice clinical reasoning by analyzing case presentations, evaluating diagnostic thinking, and guiding learners with targeted Socratic feedback.

### Team Members
- Katie Xiao: Full-Stack Development (Backend, Firebase, Frontend)
- Parvi Chadha: AI Agent Design & Prompt Engineering
- Abigail Joseph: Frontend Development & System Integration
- Caleb Kim: Bayesian Network & Backend Prompt Engineering

---

## Quick Start (Local Development)

### 1. Configure environment variables

`backend/.env`

```env
OPENAI_API_KEY=your_openai_api_key
FIREBASE_SERVICE_ACCOUNT_KEY=your_firebase_service_account_key
```

`frontend/.env`

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

Corresponding example env files can be found in `backend/.env.example` and `frontend/.env.example`.

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

## Project Structure
```text
Kaggle-MedGemma/
|-- backend/
|   |-- server.py                         # FastAPI server entrypoint + session API
|   |-- main.py                           # Local CLI tutoring entrypoint
|   |-- medgemma_client.py                # OpenAI-backed teaching brief helper
|   |-- agents/
|   |   `-- ai_attending.py               # Attending response agent
|   |-- bayes/
|   |   |-- noisy_or_bayesnet.py          # Noisy-OR Bayesian inference engine
|   |   |-- network_data.py               # Disease/symptom network definitions
|   |   |-- demo.py
|   |   `-- data/
|   |-- evaluation/
|   |   |-- presentation_workflow.py      # 9-metric rubric evaluation + question generation
|   |   `-- diagnosis_evaluator.py        # Differential support checks
|   |-- parsing/
|   |   `-- student_parser.py             # Free-text to structured clinical features
|   `-- pipeline/
|       |-- pipeline.py                   # Main orchestration: parse -> infer -> evaluate -> respond
|       `-- state.py                      # Conversation state model
|-- frontend/
|   `-- src/
|       |-- main.tsx                      # React entrypoint
|       |-- lib/
|       |   `-- firebase.ts               # Firebase client initialization
|       `-- app/
|           |-- App.tsx                   
|           `-- components/
|-- docs/
|   `-- system-diagram.png
`-- README.md
```

---

## System Diagram
![System Diagram](docs/system-diagram.png)

### Frontend 
The frontend, provides an interactive tutoring interface where students present cases and receive attending-style feedback. **Firebase Authentication** handles secure login via Google, and **Cloud Firestore** stores completed cases and the learner’s progress over time. The platform also includes **streaks, achievements, and other progress metrics** to incentivize consistent practice and directly track the student’s improvement in their clinical reasoning skills. 

### Backend
The backend runs a stateful clinical reasoning pipeline for each tutoring session.

When a student submits a case presentation, the backend works as follows:
1. An OpenAI model converts the student’s narrative into a **structured representation**, extracting normalized symptoms, absent findings, and proposed diagnoses aligned to a clinical ontology.
2. These observations update a **Noisy-OR Bayesian network**, which computes posterior probabilities across candidate pulmonary diseases and produces a ranked differential diagnosis. 
3. The student’s presentation is graded across **nine clinical reasoning competencies** allowing the system to identify missing or incomplete reasoning steps. 
4. An AI attending agent (also powered by an OpenAI model), generates concise coaching and asks **one Socratic follow-up question** that targets the student’s weakest reasoning area. 

Each interaction updates the tutoring pipeline state, which is serialized and stored in a backend **SQLite session database** to allow conversations to resume deterministically. 

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
- SQLite
- Noisy-OR Bayesian Networks

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
