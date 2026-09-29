/** Firestore triggers for the OptiCarVis EU preference service. */

const { setGlobalOptions } = require("firebase-functions/v2");
const { onDocumentCreated } = require("firebase-functions/v2/firestore");
const { onCall, HttpsError } = require("firebase-functions/v2/https");
const { defineSecret } = require("firebase-functions/params");
const { initializeApp } = require("firebase-admin/app");
const { getFirestore } = require("firebase-admin/firestore");

setGlobalOptions({ region: "europe-west1" });
initializeApp();

const OPTIMIZER_SHARED_SECRET = defineSecret("OPTIMIZER_SHARED_SECRET");
const CLOUD_RUN_URL = process.env.CLOUD_RUN_URL;

// Triggers retry until the optimiser answers. A participant on Prolific is
// waiting on the other end, so an event older than this is abandoned rather
// than retried for the platform's full seven days; the page's Retry button
// (resumePreference below) can still recover it.
const MAX_RETRY_AGE_MS = 30 * 60 * 1000;

/** Error the optimiser will keep returning; retrying cannot help. */
class PermanentOptimizerError extends Error {}

async function callOptimizer(path, payload) {
  if (!CLOUD_RUN_URL) {
    throw new Error("CLOUD_RUN_URL is missing");
  }
  // Network failures (reset connections, DNS, cold-start timeouts) throw here
  // and are retried.
  const response = await fetch(`${CLOUD_RUN_URL}${path}`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPTIMIZER_SHARED_SECRET.value()}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  const text = await response.text();
  if (response.ok) {
    console.log(`optimizer ${path}:`, text);
    return;
  }
  const message = `optimizer ${path} returned ${response.status}: ${text}`;
  // 5xx and 429 are transient (crash, overload, timeout). Any other 4xx is a
  // data or configuration error that the same request will hit again.
  if (response.status >= 500 || response.status === 429) {
    throw new Error(message);
  }
  throw new PermanentOptimizerError(message);
}

/** Run a trigger body; rethrow only what a retry can fix. */
async function withRetryPolicy(event, label, work) {
  const ageMs = Date.now() - Date.parse(event.time);
  if (ageMs > MAX_RETRY_AGE_MS) {
    console.error(`${label}: event is ${Math.round(ageMs / 1000)} s old; not retrying`);
    return;
  }
  try {
    await work();
  } catch (error) {
    if (error instanceof PermanentOptimizerError) {
      console.error(`${label}: permanent failure, not retrying: ${error.message}`);
      return;
    }
    throw error;
  }
}

exports.registerPreferenceUser = onDocumentCreated(
  {
    document: "users/{userId}",
    retry: true,
    secrets: [OPTIMIZER_SHARED_SECRET],
  },
  (event) => withRetryPolicy(event, "registerUser", () =>
    callOptimizer("/registerUser", { userId: event.params.userId })
  )
);

exports.updatePreferenceOnResult = onDocumentCreated(
  {
    document: "preferenceResults/{resultId}",
    retry: true,
    timeoutSeconds: 300,
    secrets: [OPTIMIZER_SHARED_SECRET],
  },
  (event) => withRetryPolicy(event, "updatePreference", async () => {
    const data = event.data.data();
    if (data.attentionCheckPassed === false) {
      console.log(`attention check failed for ${data.pid}; no model update`);
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
  })
);

/**
 * Called by the page when the next comparison has not arrived.
 *
 * /updatePreference is idempotent: it writes the next missing comparison (or
 * the first one, which also recovers a failed registration) and otherwise does
 * nothing, so a participant pressing Retry cannot duplicate or skip work.
 */
exports.resumePreference = onCall(
  {
    timeoutSeconds: 300,
    secrets: [OPTIMIZER_SHARED_SECRET],
  },
  async (request) => {
    if (!request.auth) {
      throw new HttpsError("unauthenticated", "Sign in first.");
    }
    const userId = request.auth.uid;
    const user = await getFirestore().collection("users").doc(userId).get();
    if (!user.exists) {
      throw new HttpsError("failed-precondition", "This session was never started.");
    }
    try {
      await callOptimizer("/updatePreference", { userId, type: "preferenceResult" });
    } catch (error) {
      console.error(`resumePreference for ${userId}: ${error.message}`);
      if (error instanceof PermanentOptimizerError) {
        throw new HttpsError("failed-precondition", "This session cannot be resumed.");
      }
      throw new HttpsError("unavailable", "The optimiser is busy. Please retry shortly.");
    }
    return { ok: true };
  }
);
