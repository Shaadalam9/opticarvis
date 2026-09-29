# OptiCarVis pairwise preference service

This folder is an independently deployable study service. It does not share the
GPU environment used by the video pipeline.

## Pilot protocol default

* Four parameters: mask alpha, trajectory alpha, background dimming, and a
  categorical palette ID.
* One forced choice question: “Which version would you prefer to have while
  riding in an automated vehicle?”
* Ten Sobol comparison pairs followed by eight EUBO comparison pairs by default.
* Eighteen comparisons in total by default. Each comparison contains options A
  and B.
* A `PairwiseGP` learns one latent preference utility.
* At the configured completion point, the evaluated configuration with the
  highest posterior mean utility is frozen for the distant city evaluation.
* Distant city responses must be written to a separate evaluation collection;
  they must never trigger `/updatePreference`.

The three psychological outcomes, clarity, perceived safety, and mental load,
belong to the final evaluation rather than the preference model.

The pilot default is 10 Sobol plus 8 EUBO comparisons following matched
synthetic sensitivity testing. Set `OPTICARVIS_PBO_EXPLORATION_COMPARISONS` and
`OPTICARVIS_PBO_EUBO_COMPARISONS` to test another deployment budget. The
effective budget and protocol identifier are frozen into each participant's
`users` document at registration. A later deployment change therefore cannot
alter an active participant's protocol.

## Firestore contract

The service writes `preferenceQueries/{pid}_comparison_N`:

```json
{
  "pid": "pseudonymous-participant-id",
  "comparisonStep": 1,
  "phase": "exploration",
  "protocolVersion": "pbo_pairwise_eubo_v3",
  "protocolId": "pbo_pairwise_eubo_v3_sobol10_eubo8",
  "comparisonBudget": {
    "explorationSobol": 10,
    "optimisationEubo": 8,
    "total": 18
  },
  "question": "Which version would you prefer to have while riding in an automated vehicle?",
  "optionA": {
    "mask_alpha": 0.14,
    "trajectory_alpha": 0.55,
    "background_dim_alpha": 0.06,
    "palette_id": 0
  },
  "optionB": {
    "mask_alpha": 0.30,
    "trajectory_alpha": 0.70,
    "background_dim_alpha": 0.10,
    "palette_id": 2
  }
}
```

The app writes `preferenceResults/{id}`:

```json
{
  "pid": "firebase-anonymous-uid",
  "comparisonStep": 1,
  "preferredOption": "prefer_a",
  "cityPhase": "familiar_optimisation",
  "attentionCheckPassed": true,
  "testMode": false,
  "clientVersion": "prolific_web_v2",
  "responseTimeMs": 4210,
  "shownAtClientMs": 1790632769527,
  "hiddenDuringComparison": false,
  "viewportWidth": 1536,
  "viewportHeight": 776
}
```

The document ID must be `{uid}_comparison_{comparisonStep}` and that
comparison must already exist in `preferenceQueries`; the rules refuse anything
else, so a client cannot answer comparisons it was never shown.
`responseTimeMs` runs from the moment the comparison is on screen, and
`hiddenDuringComparison` records whether the tab was hidden meanwhile; use both
to screen out participants who click through.

The only accepted responses are `prefer_a` and `prefer_b`. The service joins
the result to the configurations it originally wrote rather than trusting the
client to send parameter values back.

At completion it writes `studySelections/{pid}` with `selectedConfig` and
`frozenForDistantCity: true`. The selection contains the same protocol ID and
budget for auditable analysis.

## Prolific

Study URL to enter in Prolific (Prolific fills in the three placeholders):

```text
https://<hosting-domain>/?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}
```

* **Participant link.** The page refuses to start without a valid 24 character
  `PROLIFIC_PID`. `users/{uid}` stores the PID, study and session IDs plus
  screen size, viewport, device pixel ratio and coarse-pointer (touch) state.
* **One session per participant.** Starting writes
  `prolificParticipants/{PROLIFIC_PID}` and `users/{uid}` in one batch, and the
  rules accept it only while that PID is unclaimed. The same participant in a
  second browser or private window is told to continue where they started. A
  returning participant in the same browser resumes where they stopped.
* **Completion.** Set `prolificCompletionCode` in `firebase/public/study-config.js`
  (Prolific: *Study completion* → *I'll redirect them using a URL*). After the
  last comparison the page links to, and after `redirectDelaySeconds` redirects
  to, `https://app.prolific.com/submissions/complete?cc=<code>`. While the code
  is empty, participants are told to contact the researcher instead.
* **Recovery.** Both Firestore triggers retry transient optimiser failures
  (5xx, 429, network) for up to 30 minutes. Data errors return 422 and are not
  retried. If a comparison still has not arrived after 45 s, the page calls the
  `resumePreference` function once. Its Retry button calls it again; the call
  is idempotent, so it cannot duplicate or skip a comparison.
* **Testing.** Add `debug=1` and any 24 character hex PID, for example
  `?PROLIFIC_PID=aaaaaaaaaaaaaaaaaaaaaaaa&debug=1`. Debug mode shows the
  technical values and restart buttons, marks every record `testMode: true` and
  never redirects. "Start another test session" moves to a fresh random PID,
  because the old one stays claimed.

Still to provide before launch: the participant-facing page text, consent and
information sheet, the real stimulus, an attention check that follows
Prolific's policy, and the completion code.

## Local validation

Use a separate environment. Do not install these CPU service dependencies into
the OptiCarVis CUDA rendering environment.

```powershell
uv venv .venv-study --python 3.12
uv pip install --python .venv-study\Scripts\python.exe `
  --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
uv pip install --python .venv-study\Scripts\python.exe `
  -r study_service\requirements.txt
& .venv-study\Scripts\python.exe study_service\simulate.py
```

To simulate another budget, set the same variables used by the deployed
service before running the simulator:

```powershell
$env:OPTICARVIS_PBO_EXPLORATION_COMPARISONS = "10"
$env:OPTICARVIS_PBO_EUBO_COMPARISONS = "8"
& .venv-study\Scripts\python.exe study_service\simulate.py
```

`compare_budgets.py` runs matched synthetic simulations for candidate EUBO
budgets and writes an auditable JSON result under `workflow_outputs`. One seed
is the quick smoke test. Use several seeds for the actual sensitivity analysis.

```powershell
$env:OPTICARVIS_PBO_EUBO_BUDGETS = "4,8,12"
$env:OPTICARVIS_PBO_SIMULATION_SEEDS = "7"
& .venv-study\Scripts\python.exe study_service\compare_budgets.py
```

These are synthetic results, not participant evidence. The final budget must
also consider pilot completion time, fatigue, repeated choice consistency, and
selection stability.

## EU deployment

Create a separate GCP project with billing enabled. Create the Firestore
database in an EU location and keep Cloud Run and Cloud Functions in a
compatible EU region. The included Firebase triggers use `europe-west1`.

```bash
gcloud firestore databases create --location=eur3 --project <PROJECT_ID>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com cloudfunctions.googleapis.com \
  eventarc.googleapis.com firestore.googleapis.com secretmanager.googleapis.com \
  --project <PROJECT_ID>
```

Set a long random secret in both places:

* Cloud Run environment variable: `OPTICARVIS_PBO_SHARED_SECRET`
* Firebase secret: `OPTIMIZER_SHARED_SECRET`

Create the shared secret before either deployment:

```bash
cd study_service/firebase
npx firebase-tools functions:secrets:set OPTIMIZER_SHARED_SECRET \
  --project <PROJECT_ID>
cd ../..
```

Deploy the service from this directory:

```bash
gcloud run deploy opticarvis-preference \
  --source study_service \
  --region europe-west1 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 8 \
  --min-instances 1 \
  --max-instances 10 \
  --allow-unauthenticated \
  --set-env-vars FIRESTORE_DATABASE='(default)',OPTICARVIS_PBO_EXPLORATION_COMPARISONS=10,OPTICARVIS_PBO_EUBO_COMPARISONS=8 \
  --set-secrets OPTICARVIS_PBO_SHARED_SECRET=OPTIMIZER_SHARED_SECRET:latest
```

`--concurrency 8` matches the 8 gunicorn threads, so extra requests start a new
instance instead of queueing inside a busy one. `--min-instances 1` keeps one
instance warm: a cold start loads torch before the first comparison. One GP fit
plus EUBO proposal measured about 1.8 s on 2 CPU threads (about 6.8 s for the
first call after start-up). `--max-instances` caps the cost; raise it for large
simultaneous Prolific batches.

The Cloud Run ingress is public so the Firestore functions can reach it, but
the application endpoints require the shared bearer secret. The service
deliberately refuses `/registerUser` and `/updatePreference` when no shared
secret is configured. For local HTTP testing only, set
`OPTICARVIS_PBO_ALLOW_INSECURE_LOCAL=1`.

Grant the Cloud Run runtime service account Firestore access, then put the
deployed service URL in `study_service/firebase/.env`:

```text
CLOUD_RUN_URL=https://<deployed-service-url>
```

Deploy the triggers:

```bash
cd study_service/firebase
npm install
npx firebase-tools deploy --only functions,firestore:rules,firestore:indexes,hosting \
  --project <PROJECT_ID>
```

The rules, the functions and the page change together (the rules accept only
`clientVersion == "prolific_web_v2"`), so deploy all three in one command.

Before deploying `firestore.indexes.json`, merge it with all existing project
indexes. Firebase index deployment is declarative and can remove indexes that
are absent from the file.
