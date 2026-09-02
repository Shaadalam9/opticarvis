# OptiCarVis pairwise preference service

This folder is an independently deployable study service. It does not share the
GPU environment used by the video pipeline.

## Approved protocol

* Four parameters: mask alpha, trajectory alpha, background dimming, and a
  categorical palette ID.
* One forced choice question: “Which version would you prefer to have while
  riding in an automated vehicle?”
* Ten Sobol comparison pairs followed by four EUBO comparison pairs.
* Fourteen comparisons in total. Each comparison contains options A and B.
* A `PairwiseGP` learns one latent preference utility.
* After comparison 14, the evaluated configuration with the highest posterior
  mean utility is frozen for the distant city evaluation.
* Distant city responses must be written to a separate evaluation collection;
  they must never trigger `/updatePreference`.

The three psychological outcomes, clarity, perceived safety, and mental load,
belong to the final evaluation rather than the preference model.

## Firestore contract

The service writes `preferenceQueries/{pid}_comparison_N`:

```json
{
  "pid": "pseudonymous-participant-id",
  "comparisonStep": 1,
  "phase": "exploration",
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
  "pid": "pseudonymous-participant-id",
  "comparisonStep": 1,
  "preferredOption": "prefer_a",
  "cityPhase": "familiar_optimisation",
  "attentionCheckPassed": true
}
```

The only accepted responses are `prefer_a` and `prefer_b`. The service joins
the result to the configurations it originally wrote rather than trusting the
client to send parameter values back.

At completion it writes `studySelections/{pid}` with `selectedConfig` and
`frozenForDistantCity: true`.

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
  --allow-unauthenticated \
  --set-env-vars FIRESTORE_DATABASE='(default)' \
  --set-secrets OPTICARVIS_PBO_SHARED_SECRET=OPTIMIZER_SHARED_SECRET:latest
```

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
npx firebase-tools deploy --only functions,firestore:indexes \
  --project <PROJECT_ID>
```

Before deploying `firestore.indexes.json`, merge it with all existing project
indexes. Firebase index deployment is declarative and can remove indexes that
are absent from the file.
