(() => {
  "use strict";

  const DEFAULT_TOTAL = 18;
  const CLIENT_VERSION = "prolific_web_v2";
  // After this long without the next comparison, ask the backend to resume
  // once on the participant's behalf (the Firestore trigger may have failed).
  const AUTO_RESUME_MS = 45000;
  const WAIT_WARNING_MS = 120000;
  const WAIT_FAILURE_MS = 360000;
  const PROLIFIC_PID_PATTERN = /^[0-9a-f]{24}$/i;
  const views = ["loading", "start", "comparison", "complete", "error"];

  const palettes = {
    0: { target: "#FFDC00", trajectory: "#5AC8F5" },
    1: { target: "#56B4E9", trajectory: "#009E73" },
    2: { target: "#E0E0E0", trajectory: "#48CAE4" },
    3: { target: "#FF0055", trajectory: "#CCFF00" },
  };

  const studyConfig = window.OPTICARVIS_STUDY_CONFIG || {};
  const params = new URLSearchParams(window.location.search);
  // Prolific appends these to the study URL; see README "Prolific".
  const prolific = {
    pid: (params.get("PROLIFIC_PID") || "").trim(),
    studyId: (params.get("STUDY_ID") || "").trim().slice(0, 64),
    sessionId: (params.get("SESSION_ID") || "").trim().slice(0, 64),
  };
  // debug=1 shows technical values and restart buttons, marks every record
  // testMode, and never redirects to Prolific.
  const debug = params.get("debug") === "1";
  document.body.classList.toggle("debug", debug);

  const auth = firebase.auth();
  const db = firebase.firestore();
  const functions = firebase.app().functions("europe-west1");

  let currentUser = null;
  let currentQuery = null;
  let unsubscribe = null;
  let autoResume = null;
  let waitWarning = null;
  let waitFailure = null;
  let busy = false;
  let shownAtPerf = 0;
  let shownAtEpoch = 0;
  let hiddenDuringComparison = false;

  const element = (id) => document.getElementById(id);

  function showView(name) {
    views.forEach((view) => {
      element(`${view}-view`).classList.toggle("hidden", view !== name);
    });
  }

  function setLoading(message) {
    element("loading-message").textContent = message;
    showView("loading");
  }

  function clearQueryListener() {
    if (unsubscribe) {
      unsubscribe();
      unsubscribe = null;
    }
    [autoResume, waitWarning, waitFailure].forEach((timer) => window.clearTimeout(timer));
    autoResume = null;
    waitWarning = null;
    waitFailure = null;
  }

  function friendlyError(error) {
    if (error && error.code === "auth/operation-not-allowed") {
      return "Anonymous sign in is not enabled for this Firebase project.";
    }
    if (error && error.code === "permission-denied") {
      return "The study server rejected this request.";
    }
    return error && error.message ? error.message : "An unexpected error occurred.";
  }

  function showError(title, error, { canRetry = true } = {}) {
    clearQueryListener();
    element("error-title").textContent = title;
    element("error-message").textContent = friendlyError(error);
    element("retry-button").classList.toggle("hidden", !canRetry);
    showView("error");
  }

  function clampCount(value) {
    const number = Math.round(Number(value) || 0);
    return Math.min(100000, Math.max(0, number));
  }

  function viewport() {
    return {
      viewportWidth: clampCount(window.innerWidth),
      viewportHeight: clampCount(window.innerHeight),
    };
  }

  function deviceInfo() {
    return {
      screenWidth: clampCount(window.screen && window.screen.width),
      screenHeight: clampCount(window.screen && window.screen.height),
      ...viewport(),
      devicePixelRatio: Number(window.devicePixelRatio) || 1,
      pointerCoarse: window.matchMedia("(pointer: coarse)").matches,
    };
  }

  function hexToRgba(hex, alpha) {
    const clean = hex.replace("#", "");
    const value = Number.parseInt(clean, 16);
    const red = (value >> 16) & 255;
    const green = (value >> 8) & 255;
    const blue = value & 255;
    return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
  }

  function addParameter(list, label, value) {
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = label;
    description.textContent = value;
    list.append(term, description);
  }

  function renderOption(cardId, config) {
    const card = element(cardId);
    const palette = palettes[Number(config.palette_id)] || palettes[0];
    const maskAlpha = Number(config.mask_alpha);
    const trajectoryAlpha = Number(config.trajectory_alpha);
    const dimAlpha = Number(config.background_dim_alpha);

    card.style.setProperty("--dim-alpha", String(dimAlpha));
    card.style.setProperty("--target-colour", palette.target);
    card.style.setProperty("--target-glow", hexToRgba(palette.target, 0.35));
    card.style.setProperty("--mask-colour", hexToRgba(palette.target, maskAlpha));
    card.style.setProperty("--trajectory-colour", palette.trajectory);
    card.style.setProperty("--trajectory-alpha", String(trajectoryAlpha));

    const list = card.querySelector(".parameter-list");
    list.replaceChildren();
    addParameter(list, "Mask opacity", maskAlpha.toFixed(4));
    addParameter(list, "Trajectory opacity", trajectoryAlpha.toFixed(4));
    addParameter(list, "Background dimming", dimAlpha.toFixed(4));
    addParameter(list, "Palette", String(config.palette_id));
  }

  function renderQuery(query) {
    clearQueryListener();
    currentQuery = query;
    busy = false;

    const step = Number(query.comparisonStep);
    const total = Number(query.comparisonBudget?.total || DEFAULT_TOTAL);
    element("step-label").textContent = `Comparison ${step} of ${total}`;
    element("phase-label").textContent =
      query.phase === "optimisation" ? "Optimisation" : "Exploration";
    element("preference-question").textContent = query.question;

    const progress = element("progress-fill");
    progress.style.width = `${(step / total) * 100}%`;
    const track = progress.parentElement;
    track.setAttribute("aria-valuemax", String(total));
    track.setAttribute("aria-valuenow", String(step));

    renderOption("option-a", query.optionA);
    renderOption("option-b", query.optionB);
    document.querySelectorAll(".choice-button").forEach((button) => {
      button.disabled = false;
      button.textContent = button.dataset.choice === "prefer_a"
        ? "I prefer version A"
        : "I prefer version B";
    });
    showView("comparison");

    // Response time runs from the moment this comparison is on screen.
    shownAtPerf = performance.now();
    shownAtEpoch = Date.now();
    hiddenDuringComparison = document.visibilityState === "hidden";
  }

  document.addEventListener("visibilitychange", () => {
    if (currentQuery && document.visibilityState === "hidden") {
      hiddenDuringComparison = true;
    }
  });

  /** Ask the backend to write whatever the participant is waiting for. */
  async function requestResume() {
    await functions.httpsCallable("resumePreference")();
  }

  function scheduleWaitTimers(failureTitle) {
    autoResume = window.setTimeout(() => {
      requestResume().catch((error) => console.warn("automatic resume failed", error));
    }, AUTO_RESUME_MS);
    waitWarning = window.setTimeout(() => {
      element("loading-message").textContent =
        "This is taking longer than usual. Please keep this page open.";
    }, WAIT_WARNING_MS);
    waitFailure = window.setTimeout(() => {
      showError(
        failureTitle,
        new Error("Please press Retry. Your earlier answers are saved and will not be repeated.")
      );
    }, WAIT_FAILURE_MS);
  }

  async function waitForSelection() {
    clearQueryListener();
    setLoading("Saving your answers…");
    const selectionRef = db.collection("studySelections").doc(currentUser.uid);
    unsubscribe = selectionRef.onSnapshot(
      (snapshot) => {
        if (snapshot.exists) {
          renderCompletion(snapshot.data());
        }
      },
      (error) => showError("Your answers could not be finalised", error)
    );
    scheduleWaitTimers("Finalising is taking longer than expected");
  }

  function waitForQuery(step) {
    clearQueryListener();
    setLoading(step === 1 ? "Preparing the first comparison…" : `Preparing comparison ${step}…`);
    const queryId = `${currentUser.uid}_comparison_${step}`;
    const queryRef = db.collection("preferenceQueries").doc(queryId);
    unsubscribe = queryRef.onSnapshot(
      (snapshot) => {
        if (snapshot.exists) {
          renderQuery(snapshot.data());
        }
      },
      (error) => showError("The next comparison could not be loaded", error)
    );
    scheduleWaitTimers("The next comparison is taking longer than expected");
  }

  function completionUrl() {
    const code = String(studyConfig.prolificCompletionCode || "").trim();
    if (!code) {
      return "";
    }
    const base = studyConfig.prolificCompletionBaseUrl
      || "https://app.prolific.com/submissions/complete?cc=";
    return base + encodeURIComponent(code);
  }

  function renderCompletion(selection) {
    clearQueryListener();
    const container = element("selected-config");
    container.replaceChildren();
    const config = selection.selectedConfig || {};
    const total = Number(selection.comparisonBudget?.total || DEFAULT_TOTAL);
    element("complete-title").textContent = `All ${total} comparisons are complete`;
    [
      ["Mask opacity", config.mask_alpha],
      ["Trajectory opacity", config.trajectory_alpha],
      ["Background dimming", config.background_dim_alpha],
      ["Palette", config.palette_id],
    ].forEach(([label, value]) => {
      const item = document.createElement("div");
      const name = document.createElement("span");
      const output = document.createElement("strong");
      name.textContent = label;
      output.textContent = value === undefined ? "Not available" : String(value);
      item.append(name, output);
      container.append(item);
    });

    const url = completionUrl();
    const link = element("prolific-link");
    const message = element("complete-message");
    if (!url) {
      link.classList.add("hidden");
      message.textContent =
        "Thank you. The completion code is not configured yet; please contact the researcher "
        + "through Prolific so your submission can be credited.";
    } else {
      link.href = url;
      link.classList.remove("hidden");
      const delay = Math.max(0, Number(studyConfig.redirectDelaySeconds ?? 5));
      if (debug) {
        message.textContent = "Debug mode: the Prolific redirect is shown but not followed.";
      } else {
        message.textContent =
          `Thank you. You will be returned to Prolific in ${delay} seconds.`;
        window.setTimeout(() => window.location.assign(url), delay * 1000);
      }
    }
    showView("complete");
  }

  async function completedSteps(total) {
    let completed = 0;
    for (let step = 1; step <= total; step += 1) {
      const resultId = `${currentUser.uid}_comparison_${step}`;
      const snapshot = await db.collection("preferenceResults").doc(resultId).get();
      if (!snapshot.exists) {
        break;
      }
      completed = step;
    }
    return completed;
  }

  async function resumeSession() {
    setLoading("Restoring your session…");
    const selection = await db.collection("studySelections").doc(currentUser.uid).get();
    if (selection.exists) {
      renderCompletion(selection.data());
      return;
    }

    const user = await db.collection("users").doc(currentUser.uid).get();
    const total = Number(
      user.data()?.preferenceProtocol?.comparisonBudget?.total || DEFAULT_TOTAL
    );
    const completed = await completedSteps(total);
    if (completed >= total) {
      await waitForSelection();
      return;
    }
    waitForQuery(completed + 1);
  }

  async function startSession() {
    if (!currentUser || busy) {
      return;
    }
    busy = true;
    element("start-button").disabled = true;
    setLoading("Starting your session…");
    try {
      // The Prolific claim and the user record are written together; the
      // rules accept the pair only if the PID has never been claimed, so the
      // same participant cannot start a second session elsewhere.
      const createdAt = firebase.firestore.FieldValue.serverTimestamp();
      const batch = db.batch();
      batch.set(db.collection("prolificParticipants").doc(prolific.pid), {
        uid: currentUser.uid,
        studyId: prolific.studyId,
        sessionId: prolific.sessionId,
        createdAt,
      });
      batch.set(db.collection("users").doc(currentUser.uid), {
        createdAt,
        testMode: debug,
        clientVersion: CLIENT_VERSION,
        prolificPid: prolific.pid,
        prolificStudyId: prolific.studyId,
        prolificSessionId: prolific.sessionId,
        ...deviceInfo(),
      });
      await batch.commit();
      await resumeSession();
    } catch (error) {
      busy = false;
      element("start-button").disabled = false;
      if (error && error.code === "permission-denied") {
        showError(
          "This Prolific ID already has a session",
          new Error(
            "The study was already started in another browser or window. Please continue "
            + "there, or contact the researcher through Prolific."
          ),
          { canRetry: false }
        );
        return;
      }
      showError("Your session could not be started", error);
    }
  }

  async function submitChoice(preferredOption) {
    if (!currentUser || !currentQuery || busy) {
      return;
    }
    busy = true;
    document.querySelectorAll(".choice-button").forEach((button) => {
      button.disabled = true;
      button.textContent = button.dataset.choice === preferredOption
        ? "Saving your choice…"
        : button.textContent;
    });

    const step = Number(currentQuery.comparisonStep);
    const total = Number(currentQuery.comparisonBudget?.total || DEFAULT_TOTAL);
    const resultId = `${currentUser.uid}_comparison_${step}`;
    try {
      await db.collection("preferenceResults").doc(resultId).set({
        pid: currentUser.uid,
        comparisonStep: step,
        preferredOption,
        cityPhase: "familiar_optimisation",
        attentionCheckPassed: true,
        submittedAt: firebase.firestore.FieldValue.serverTimestamp(),
        testMode: debug,
        clientVersion: CLIENT_VERSION,
        responseTimeMs: Math.max(0, Math.round(performance.now() - shownAtPerf)),
        shownAtClientMs: Math.round(shownAtEpoch),
        hiddenDuringComparison,
        ...viewport(),
      });
      currentQuery = null;
      if (step >= total) {
        await waitForSelection();
      } else {
        waitForQuery(step + 1);
      }
    } catch (error) {
      busy = false;
      showError("Your choice could not be saved", error);
    }
  }

  function randomTestPid() {
    const bytes = new Uint8Array(12);
    window.crypto.getRandomValues(bytes);
    return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
  }

  /** Debug only: a fresh anonymous session under a fresh test PID. */
  async function newSession() {
    if (!debug) {
      return;
    }
    clearQueryListener();
    setLoading("Creating a new test session…");
    try {
      await auth.signOut();
      // The old PID stays claimed by the old session, so reuse would be refused.
      params.set("PROLIFIC_PID", randomTestPid());
      window.location.search = params.toString();
    } catch (error) {
      showError("A new session could not be created", error);
    }
  }

  element("start-button").addEventListener("click", startSession);
  element("new-session-button").addEventListener("click", newSession);
  element("reset-button").addEventListener("click", newSession);
  element("retry-button").addEventListener("click", async () => {
    if (!currentUser) {
      window.location.reload();
      return;
    }
    setLoading("Retrying…");
    try {
      // Re-request the missing comparison, then listen for it again.
      await requestResume();
    } catch (error) {
      console.warn("resume request failed", error);
    }
    resumeSession().catch((error) => showError("Your session could not be resumed", error));
  });
  document.querySelectorAll(".choice-button").forEach((button) => {
    button.addEventListener("click", () => submitChoice(button.dataset.choice));
  });

  if (!PROLIFIC_PID_PATTERN.test(prolific.pid)) {
    showError(
      "Please open this study from Prolific",
      new Error(
        "This link is missing your Prolific ID. Return to Prolific and open the study from there."
      ),
      { canRetry: false }
    );
    return;
  }

  auth.onAuthStateChanged(async (user) => {
    if (!user) {
      setLoading("Connecting to the study server…");
      try {
        await auth.signInAnonymously();
      } catch (error) {
        showError("Could not connect to the study server", error);
      }
      return;
    }

    currentUser = user;
    busy = false;
    element("session-label").textContent = `Session ${user.uid.slice(0, 8)}`;
    try {
      const userDocument = await db.collection("users").doc(user.uid).get();
      if (!userDocument.exists) {
        element("start-button").disabled = false;
        showView("start");
        return;
      }
      if (userDocument.data().prolificPid !== prolific.pid) {
        showError(
          "This browser already has a session for another Prolific ID",
          new Error("Please contact the researcher through Prolific."),
          { canRetry: false }
        );
        return;
      }
      await resumeSession();
    } catch (error) {
      showError("Your session could not be checked", error);
    }
  });
})();
