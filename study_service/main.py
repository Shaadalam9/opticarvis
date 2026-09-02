"""Firestore mediated OptiCarVis pairwise preference service.

The participant app writes one binary choice per comparison. The deployment
defines a default Sobol and EUBO budget, which is frozen into each participant's
Firestore record at registration. Changing the deployment default therefore
cannot alter an active participant's protocol. At completion the best evaluated
configuration is frozen for the distant city evaluation; distant city data
never updates the preference model.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import defaultdict

from flask import Flask, jsonify, request
from google.api_core.exceptions import AlreadyExists
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

import optimizer_core
import preference_data
import space


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
app = Flask(__name__)

FIRESTORE_DATABASE = os.environ.get("FIRESTORE_DATABASE", "(default)")
QUERY_COLLECTION = os.environ.get("PREFERENCE_QUERY_COLLECTION", "preferenceQueries")
RESULT_COLLECTION = os.environ.get("PREFERENCE_RESULT_COLLECTION", "preferenceResults")
SELECTION_COLLECTION = os.environ.get("PREFERENCE_SELECTION_COLLECTION", "studySelections")
USER_COLLECTION = os.environ.get("PREFERENCE_USER_COLLECTION", "users")

DEFAULT_BUDGET = space.comparison_budget_from_environment()

_db = None
_user_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)


def get_db():
    global _db
    if _db is None:
        _db = firestore.Client(database=FIRESTORE_DATABASE)
    return _db


def _stable_seed(user_id: str) -> int:
    digest = hashlib.sha256(user_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _authorised() -> bool:
    expected = os.environ.get("OPTICARVIS_PBO_SHARED_SECRET", "")
    if not expected:
        return os.environ.get("OPTICARVIS_PBO_ALLOW_INSECURE_LOCAL", "0") == "1"
    header = request.headers.get("Authorization", "")
    return header == f"Bearer {expected}"


def _unauthorised_response():
    return jsonify({"ok": False, "error": "unauthorised"}), 401


def _documents(collection_name: str, user_id: str):
    docs = (
        get_db()
        .collection(collection_name)
        .where(filter=FieldFilter("pid", "==", user_id))
        .order_by("comparisonStep")
        .stream()
    )
    return [doc.to_dict() for doc in docs]


def load_training_data(user_id: str) -> preference_data.TrainingData:
    return preference_data.build_training_data(
        _documents(QUERY_COLLECTION, user_id),
        _documents(RESULT_COLLECTION, user_id),
    )


def _query_reference(user_id: str, comparison_step: int):
    return get_db().collection(QUERY_COLLECTION).document(
        f"{user_id}_comparison_{comparison_step}"
    )


def _user_reference(user_id: str):
    return get_db().collection(USER_COLLECTION).document(user_id)


def load_or_freeze_participant_budget(user_id: str) -> space.ComparisonBudget:
    """Return the budget permanently assigned to one participant."""
    reference = _user_reference(user_id)
    snapshot = reference.get()
    document = snapshot.to_dict() if snapshot.exists else {}
    protocol = (document or {}).get("preferenceProtocol")
    if protocol is not None:
        if protocol.get("protocolVersion") != space.PROTOCOL_VERSION:
            raise ValueError(
                f"participant {user_id} uses unsupported preference protocol "
                f"{protocol.get('protocolVersion')!r}"
            )
        return space.comparison_budget_from_document(protocol["comparisonBudget"])

    protocol = space.protocol_document(DEFAULT_BUDGET)
    reference.set({"preferenceProtocol": protocol}, merge=True)
    return DEFAULT_BUDGET


def write_query(
    user_id: str,
    comparison_step: int,
    option_a: dict,
    option_b: dict,
    phase: str,
    budget: space.ComparisonBudget,
) -> bool:
    # Randomise presentation side deterministically so retries reproduce the
    # same query while A/B position cannot become a systematic confound.
    if _stable_seed(f"{user_id}:{comparison_step}:presentation") % 2:
        option_a, option_b = option_b, option_a
    document = {
        "pid": user_id,
        "comparisonStep": comparison_step,
        "phase": phase,
        "question": space.PREFERENCE_QUESTION,
        "optionA": space.validate_config(option_a),
        "optionB": space.validate_config(option_b),
        "cityPhase": "familiar_optimisation",
        "presentationOrderRandomised": True,
        "createdAt": firestore.SERVER_TIMESTAMP,
        **space.protocol_document(budget),
    }
    try:
        _query_reference(user_id, comparison_step).create(document)
        return True
    except AlreadyExists:
        log.warning(
            "comparison %d already exists for %s; duplicate suppressed",
            comparison_step,
            user_id,
        )
        return False


def finalize_participant(
    user_id: str,
    training: preference_data.TrainingData,
    budget: space.ComparisonBudget,
):
    model = optimizer_core.fit_preference_model(training)
    final_config, posterior_mean = optimizer_core.select_best_observed(
        model, training.raw_configs, training.model_rows
    )
    selection = {
        "pid": user_id,
        "selectedConfig": final_config,
        "selectionRule": "highest_posterior_mean_among_evaluated_configs",
        "latentUtilityPosteriorMean": posterior_mean,
        "comparisonsCompleted": len(training.comparisons),
        "frozenForDistantCity": True,
        "modelUpdatedByDistantCity": False,
        "createdAt": firestore.SERVER_TIMESTAMP,
        **space.protocol_document(budget),
    }
    get_db().collection(SELECTION_COLLECTION).document(user_id).set(selection, merge=True)
    _user_reference(user_id).set(
        {
            "studyOptimisationCompleted": True,
            "studyOptimisationCompletedAt": firestore.SERVER_TIMESTAMP,
            "completedPreferenceProtocol": space.protocol_document(budget),
        },
        merge=True,
    )
    return selection


@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "method": "PairwiseGP_EUBO",
            "preferenceQuestion": space.PREFERENCE_QUESTION,
            "parameters": space.PARAMETER_NAMES,
            "modelDimensions": space.D_MODEL,
            "protocolVersion": space.PROTOCOL_VERSION,
            "defaultProtocolId": DEFAULT_BUDGET.protocol_id,
            "defaultComparisonBudget": DEFAULT_BUDGET.to_document(),
            "database": FIRESTORE_DATABASE,
        }
    )


@app.post("/registerUser")
def register_user():
    if not _authorised():
        return _unauthorised_response()
    payload = request.get_json(force=True)
    user_id = str(payload.get("userId", "")).strip()
    if not user_id:
        return jsonify({"ok": False, "error": "missing userId"}), 400

    with _user_locks[user_id]:
        budget = load_or_freeze_participant_budget(user_id)
        if _query_reference(user_id, 1).get().exists:
            return jsonify(
                {
                    "ok": True,
                    "skipped": True,
                    "protocolId": budget.protocol_id,
                    "comparisonBudget": budget.to_document(),
                }
            )
        option_a, option_b = optimizer_core.sobol_pair(1, seed=_stable_seed(user_id))
        write_query(user_id, 1, option_a, option_b, "exploration", budget)
        return jsonify(
            {
                "ok": True,
                "comparisonStep": 1,
                "protocolId": budget.protocol_id,
                "comparisonBudget": budget.to_document(),
            }
        )


@app.post("/updatePreference")
def update_preference():
    if not _authorised():
        return _unauthorised_response()
    payload = request.get_json(force=True)
    if payload.get("type") not in (None, "preferenceResult"):
        return jsonify({"ok": True, "ignored": True})
    user_id = str(payload.get("userId", "")).strip()
    if not user_id:
        return jsonify({"ok": False, "error": "missing userId"}), 400

    with _user_locks[user_id]:
        budget = load_or_freeze_participant_budget(user_id)
        training = load_training_data(user_id)
        completed = len(training.comparisons)
        if completed >= budget.total_comparisons:
            selection = finalize_participant(user_id, training, budget)
            return jsonify(
                {
                    "ok": True,
                    "studyCompleted": True,
                    "selectedConfig": selection["selectedConfig"],
                }
            )

        next_step = completed + 1
        if _query_reference(user_id, next_step).get().exists:
            return jsonify({"ok": True, "skipped": True, "nextComparisonStep": next_step})

        seed = _stable_seed(user_id)
        if completed < budget.exploration_comparisons:
            option_a, option_b = optimizer_core.sobol_pair(next_step, seed=seed)
            phase = "exploration"
        else:
            model = optimizer_core.fit_preference_model(training)
            option_a, option_b = optimizer_core.propose_eubo_pair(
                model,
                observed_pair_keys=training.observed_pair_keys,
                comparison_step=next_step,
                seed=seed + next_step,
            )
            phase = "optimisation"

        write_query(user_id, next_step, option_a, option_b, phase, budget)
        return jsonify(
            {
                "ok": True,
                "phase": phase,
                "comparisonsCompleted": completed,
                "nextComparisonStep": next_step,
                "protocolId": budget.protocol_id,
                "comparisonBudget": budget.to_document(),
            }
        )
