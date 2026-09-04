"""HTTP lifecycle test against an in memory Firestore replacement."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass


SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SERVICE_ROOT)

os.environ["OPTICARVIS_PBO_ALLOW_INSECURE_LOCAL"] = "1"

import main
import space


@dataclass
class FakeSnapshot:
    reference: object
    data: dict | None

    @property
    def exists(self):
        return self.data is not None

    def to_dict(self):
        return dict(self.data or {})


class FakeDocumentReference:
    def __init__(self, collection, document_id):
        self.collection = collection
        self.document_id = document_id

    def get(self):
        return FakeSnapshot(self, self.collection.documents.get(self.document_id))

    def create(self, data):
        if self.document_id in self.collection.documents:
            raise RuntimeError("document already exists")
        self.collection.documents[self.document_id] = dict(data)

    def set(self, data, merge=False):
        if merge:
            current = dict(self.collection.documents.get(self.document_id, {}))
            current.update(data)
            self.collection.documents[self.document_id] = current
        else:
            self.collection.documents[self.document_id] = dict(data)


class FakeQuery:
    def __init__(self, collection, participant_id=None, order_field=None):
        self.collection = collection
        self.participant_id = participant_id
        self.order_field = order_field

    def where(self, filter):
        return FakeQuery(self.collection, filter.value, self.order_field)

    def order_by(self, field):
        return FakeQuery(self.collection, self.participant_id, field)

    def stream(self):
        rows = [
            (document_id, data)
            for document_id, data in self.collection.documents.items()
            if self.participant_id is None or data.get("pid") == self.participant_id
        ]
        if self.order_field:
            rows.sort(key=lambda item: item[1].get(self.order_field, 0))
        return [
            FakeSnapshot(FakeDocumentReference(self.collection, document_id), data)
            for document_id, data in rows
        ]


class FakeCollection(FakeQuery):
    def __init__(self):
        self.documents = {}
        super().__init__(self)

    def document(self, document_id):
        return FakeDocumentReference(self, document_id)


class FakeFirestore:
    def __init__(self):
        self.collections = {}

    def collection(self, name):
        if name not in self.collections:
            self.collections[name] = FakeCollection()
        return self.collections[name]


class FakeModel:
    pass


def _adaptive_pair(comparison_step):
    base = space.default_config()
    offset = comparison_step
    option_a = dict(
        base,
        mask_alpha=min(0.7, 0.1 * offset),
        palette_id=offset % 4,
    )
    option_b = dict(
        base,
        trajectory_alpha=min(1.0, 0.2 * offset),
        palette_id=(offset + 1) % 4,
    )
    return option_a, option_b


def test_complete_eighteen_comparison_lifecycle():
    fake_db = FakeFirestore()
    main._db = fake_db
    main.DEFAULT_BUDGET = space.ComparisonBudget(10, 8)
    main.optimizer_core.fit_preference_model = lambda training: FakeModel()
    main.optimizer_core.propose_eubo_pair = (
        lambda model, observed_pair_keys, comparison_step, seed: _adaptive_pair(
            comparison_step
        )
    )
    main.optimizer_core.select_best_observed = (
        lambda model, raw_configs, model_rows: (raw_configs[-1], 1.25)
    )

    client = main.app.test_client()
    response = client.post("/registerUser", json={"userId": "participant-1"})
    assert response.status_code == 200
    assert response.get_json()["comparisonBudget"]["total"] == 18
    user = fake_db.collection(main.USER_COLLECTION).documents["participant-1"]
    assert user["preferenceProtocol"]["protocolId"].endswith("_sobol10_eubo8")

    for step in range(1, 19):
        query_id = f"participant-1_comparison_{step}"
        queries = fake_db.collection(main.QUERY_COLLECTION).documents
        assert query_id in queries
        query = queries[query_id]
        assert query["comparisonStep"] == step
        assert query["presentationOrderRandomised"] is True
        assert query["comparisonBudget"]["total"] == 18

        fake_db.collection(main.RESULT_COLLECTION).document(f"result-{step}").create(
            {
                "pid": "participant-1",
                "comparisonStep": step,
                "preferredOption": "prefer_a",
                "cityPhase": "familiar_optimisation",
                "attentionCheckPassed": True,
            }
        )
        response = client.post(
            "/updatePreference",
            json={"userId": "participant-1", "type": "preferenceResult"},
        )
        assert response.status_code == 200, response.data

    result = response.get_json()
    assert result["studyCompleted"] is True
    assert len(fake_db.collection(main.QUERY_COLLECTION).documents) == 18
    selection = fake_db.collection(main.SELECTION_COLLECTION).documents["participant-1"]
    assert selection["comparisonsCompleted"] == 18
    assert selection["frozenForDistantCity"] is True
    assert selection["modelUpdatedByDistantCity"] is False
    assert selection["comparisonBudget"]["total"] == 18


def test_participant_budget_is_frozen_at_registration():
    fake_db = FakeFirestore()
    main._db = fake_db
    original_default = main.DEFAULT_BUDGET
    main.DEFAULT_BUDGET = space.ComparisonBudget(2, 2)
    main.optimizer_core.fit_preference_model = lambda training: FakeModel()
    main.optimizer_core.propose_eubo_pair = (
        lambda model, observed_pair_keys, comparison_step, seed: _adaptive_pair(
            comparison_step
        )
    )
    main.optimizer_core.select_best_observed = (
        lambda model, raw_configs, model_rows: (raw_configs[-1], 1.25)
    )

    try:
        client = main.app.test_client()
        response = client.post("/registerUser", json={"userId": "participant-frozen"})
        assert response.status_code == 200
        assert response.get_json()["comparisonBudget"]["total"] == 4

        # A later deployment default must not change this active participant.
        main.DEFAULT_BUDGET = space.ComparisonBudget(10, 8)

        for step in range(1, 5):
            fake_db.collection(main.RESULT_COLLECTION).document(f"frozen-{step}").create(
                {
                    "pid": "participant-frozen",
                    "comparisonStep": step,
                    "preferredOption": "prefer_a",
                    "cityPhase": "familiar_optimisation",
                    "attentionCheckPassed": True,
                }
            )
            response = client.post(
                "/updatePreference",
                json={"userId": "participant-frozen", "type": "preferenceResult"},
            )
            assert response.status_code == 200, response.data

        assert response.get_json()["studyCompleted"] is True
        queries = fake_db.collection(main.QUERY_COLLECTION).documents
        assert "participant-frozen_comparison_5" not in queries
        assert queries["participant-frozen_comparison_2"]["phase"] == "exploration"
        assert queries["participant-frozen_comparison_3"]["phase"] == "optimisation"
        selection = fake_db.collection(main.SELECTION_COLLECTION).documents[
            "participant-frozen"
        ]
        assert selection["comparisonBudget"] == {
            "explorationSobol": 2,
            "optimisationEubo": 2,
            "total": 4,
        }
    finally:
        main.DEFAULT_BUDGET = original_default


def main_test_runner():
    tests = [
        test_complete_eighteen_comparison_lifecycle,
        test_participant_budget_is_frozen_at_registration,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main_test_runner()
