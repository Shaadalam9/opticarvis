/*
 * Per-deployment study settings. Edit this file, then redeploy hosting.
 *
 * prolificCompletionCode: the completion code from the Prolific study page
 *   ("Study completion" -> "I'll redirect them using a URL"). While it is
 *   empty, finishing participants are told the code is missing instead of
 *   being redirected, so a deployment without it cannot silently lose
 *   submissions.
 */
window.OPTICARVIS_STUDY_CONFIG = Object.freeze({
  prolificCompletionCode: "",
  prolificCompletionBaseUrl: "https://app.prolific.com/submissions/complete?cc=",
  // Seconds before the automatic redirect to Prolific after the last comparison.
  redirectDelaySeconds: 5,
});
