/** Firestore triggers for the OptiCarVis EU preference service. */

const { setGlobalOptions } = require("firebase-functions/v2");
const { onDocumentCreated } = require("firebase-functions/v2/firestore");
const { defineSecret } = require("firebase-functions/params");
const { initializeApp } = require("firebase-admin/app");

setGlobalOptions({ region: "europe-west1" });
initializeApp();

const OPTIMIZER_SHARED_SECRET = defineSecret("OPTIMIZER_SHARED_SECRET");
const CLOUD_RUN_URL = process.env.CLOUD_RUN_URL;

async function callOptimizer(path, payload) {
  if (!CLOUD_RUN_URL) {
    throw new Error("CLOUD_RUN_URL is missing");
  }
  const response = await fetch(`${CLOUD_RUN_URL}${path}`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPTIMIZER_SHARED_SECRET.value()}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`optimizer ${path} returned ${response.status}: ${text}`);
  }
  console.log(`optimizer ${path}:`, text);
}

exports.registerPreferenceUser = onDocumentCreated(
  {
    document: "users/{userId}",
    secrets: [OPTIMIZER_SHARED_SECRET],
  },
  async (event) => {
    await callOptimizer("/registerUser", { userId: event.params.userId });
  }
);

exports.updatePreferenceOnResult = onDocumentCreated(
  {
    document: "preferenceResults/{resultId}",
    timeoutSeconds: 300,
    secrets: [OPTIMIZER_SHARED_SECRET],
  },
  async (event) => {
    const data = event.data.data();
    if (data.attentionCheckPassed === false) {
      console.log(`attention check failed for ${data.pid}; comparison is repeated`);
      return;
    }
    if (data.cityPhase !== "familiar_optimisation") {
      console.log(`ignoring non-optimisation result for ${data.pid}`);
      return;
    }
    await callOptimizer("/updatePreference", {
      userId: data.pid,
      type: "preferenceResult",
      comparisonStep: data.comparisonStep,
    });
  }
);
