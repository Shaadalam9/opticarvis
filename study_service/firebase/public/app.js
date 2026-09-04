(() => {
  "use strict";

  const DEFAULT_TOTAL = 18;
  const CLIENT_VERSION = "internal_web_v1";
  const WAIT_WARNING_MS = 120000;
  const WAIT_FAILURE_MS = 360000;
  const views = ["loading", "start", "comparison", "complete", "error"];

  const palettes = {
    0: { target: "#FF9F1C", trajectory: "#00B4D8" },
    1: { target: "#56B4E9", trajectory: "#009E73" },
    2: { target: "#E0E0E0", trajectory: "#48CAE4" },
    3: { target: "#FF0055", trajectory: "#CCFF00" },
  };

  const auth = firebase.auth();
  const db = firebase.firestore();

  let currentUser = null;
  let currentQuery = null;
  let unsubscribe = null;
  let waitWarning = null;
  let waitFailure = null;
  let busy = false;

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
    window.clearTimeout(waitWarning);
    window.clearTimeout(waitFailure);
    waitWarning = null;
    waitFailure = null;
  }

  function friendlyError(error) {
    if (error && error.code === "auth/operation-not-allowed") {
      return "Anonymous sign in is not enabled for this Firebase project.";
    }
    if (error && error.code === "permission-denied") {
      return "Firestore rejected this request. Check that the test rules were deployed.";
    }
    return error && error.message ? error.message : "An unexpected error occurred.";
  }

  function showError(title, error) {
    clearQueryListener();
    element("error-title").textContent = title;
    element("error-message").textContent = friendlyError(error);
    showView("error");
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
  }

  async function waitForSelection() {
    clearQueryListener();
    setLoading("Finalising the selected configuration…");
    const selectionRef = db.collection("studySelections").doc(currentUser.uid);
    unsubscribe = selectionRef.onSnapshot(
      (snapshot) => {
        if (snapshot.exists) {
          renderCompletion(snapshot.data());
        }
      },
      (error) => showError("The final selection could not be loaded", error)
    );
    waitFailure = window.setTimeout(() => {
      showError(
        "The final selection is taking longer than expected",
        new Error("Please retry. The submitted comparison is safe and will not be duplicated.")
      );
    }, WAIT_FAILURE_MS);
  }

  function waitForQuery(step) {
    clearQueryListener();
    setLoading(
      step === 1
        ? "Generating the first comparison…"
        : `Generating comparison ${step}…`
    );
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
    waitWarning = window.setTimeout(() => {
      element("loading-message").textContent =
        "The optimiser is still preparing the next comparison. You can keep this page open.";
    }, WAIT_WARNING_MS);
    waitFailure = window.setTimeout(() => {
      showError(
        "The optimiser is taking longer than expected",
        new Error("Please retry. Existing responses will be resumed automatically.")
      );
    }, WAIT_FAILURE_MS);
  }

  function renderCompletion(selection) {
    clearQueryListener();
    const container = element("selected-config");
    container.replaceChildren();
    const config = selection.selectedConfig || {};
    const total = Number(selection.comparisonBudget?.total || DEFAULT_TOTAL);
    element("complete-title").textContent = `All ${total} comparisons were processed`;
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
    setLoading("Restoring the test session…");
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
    setLoading("Registering the test session…");
    try {
      const userRef = db.collection("users").doc(currentUser.uid);
      await db.runTransaction(async (transaction) => {
        const snapshot = await transaction.get(userRef);
        if (!snapshot.exists) {
          transaction.set(userRef, {
            createdAt: firebase.firestore.FieldValue.serverTimestamp(),
            testMode: true,
            clientVersion: CLIENT_VERSION,
          });
        }
      });
      await resumeSession();
    } catch (error) {
      busy = false;
      element("start-button").disabled = false;
      showError("The test session could not be started", error);
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
        ? "Submitting choice…"
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
        testMode: true,
        clientVersion: CLIENT_VERSION,
      });
      currentQuery = null;
      if (step >= total) {
        await waitForSelection();
      } else {
        waitForQuery(step + 1);
      }
    } catch (error) {
      busy = false;
      showError("The preference could not be submitted", error);
    }
  }

  async function newSession() {
    clearQueryListener();
    setLoading("Creating a new anonymous test session…");
    try {
      await auth.signOut();
    } catch (error) {
      showError("A new test session could not be created", error);
    }
  }

  element("start-button").addEventListener("click", startSession);
  element("new-session-button").addEventListener("click", newSession);
  element("reset-button").addEventListener("click", newSession);
  element("retry-button").addEventListener("click", () => {
    if (currentUser) {
      resumeSession().catch((error) => showError("The session could not be resumed", error));
    } else {
      window.location.reload();
    }
  });
  document.querySelectorAll(".choice-button").forEach((button) => {
    button.addEventListener("click", () => submitChoice(button.dataset.choice));
  });

  auth.onAuthStateChanged(async (user) => {
    if (!user) {
      setLoading("Creating a secure anonymous test session…");
      try {
        await auth.signInAnonymously();
      } catch (error) {
        showError("Firebase sign in failed", error);
      }
      return;
    }

    currentUser = user;
    busy = false;
    element("session-label").textContent = `Session ${user.uid.slice(0, 8)}`;
    try {
      const userDocument = await db.collection("users").doc(user.uid).get();
      if (userDocument.exists) {
        await resumeSession();
      } else {
        element("start-button").disabled = false;
        showView("start");
      }
    } catch (error) {
      showError("The test session could not be checked", error);
    }
  });
})();
