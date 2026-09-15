"""Unit tests for NLP bottleneck proxies (Spec Section 2.9) and SHAP driver schemas."""

import pytest

from app.paimana_nlp import NLP_BOTTLENECK_RULES, extract_bottleneck_proxies
from app.paimana_shap import FEATURE_POLICY_TRANSLATION, explain_prediction_shap


class TestBottleneckProxies:
    def test_all_five_ontology_tags(self):
        assert set(NLP_BOTTLENECK_RULES.keys()) == {
            "LAND_ROW", "ENV_FOREST", "LEGAL_CONTRACTOR",
            "UTILITY_INTERAGENCY", "LOCAL_GEOLOGY",
        }

    def test_none_remarks_all_zero(self):
        assert extract_bottleneck_proxies(None) == {k: 0 for k in NLP_BOTTLENECK_RULES}

    def test_empty_remarks_all_zero(self):
        assert all(v == 0 for v in extract_bottleneck_proxies("").values())

    def test_non_string_remarks_all_zero(self):
        assert all(v == 0 for v in extract_bottleneck_proxies(12345).values())

    def test_land_row_detection(self):
        r = extract_bottleneck_proxies("Land acquisition compensation pending with SLAO")
        assert r["LAND_ROW"] == 1
        assert sum(r.values()) == 1

    def test_forest_clearance_detection(self):
        r = extract_bottleneck_proxies("Stage-II forest clearance awaited from MoEFCC")
        assert r["ENV_FOREST"] == 1

    def test_arbitration_detection(self):
        r = extract_bottleneck_proxies("Matter referred to arbitration tribunal; contractor dispute")
        assert r["LEGAL_CONTRACTOR"] == 1

    def test_utility_shifting_detection(self):
        r = extract_bottleneck_proxies("Utility shifting of 220kV transmission line pending; CRS approval awaited")
        assert r["UTILITY_INTERAGENCY"] == 1

    def test_geology_detection(self):
        r = extract_bottleneck_proxies("Flash flood damaged approach road; landslide in cutting zone")
        assert r["LOCAL_GEOLOGY"] == 1

    def test_multi_tag_remark(self):
        r = extract_bottleneck_proxies(
            "Right of way handover delayed by encroachment; forest clearance and arbitration ongoing"
        )
        assert r["LAND_ROW"] == 1
        assert r["ENV_FOREST"] == 1
        assert r["LEGAL_CONTRACTOR"] == 1

    def test_case_insensitive(self):
        r = extract_bottleneck_proxies("FOREST CLEARANCE PENDING")
        assert r["ENV_FOREST"] == 1

    def test_word_boundaries_no_false_positive(self):
        # 'row' must not match inside 'crowd' or 'grow'
        r = extract_bottleneck_proxies("Crowd management and plan growth activities proceeding")
        assert r["LAND_ROW"] == 0


class TestSHAPTranslationCatalog:
    def test_known_leakage_feature_documented(self):
        assert "has_revised_doc" in FEATURE_POLICY_TRANSLATION
        assert "RISK_INCREASING" in FEATURE_POLICY_TRANSLATION["has_revised_doc"]

    def test_translations_are_direction_complete(self):
        for feat, dirs in FEATURE_POLICY_TRANSLATION.items():
            assert set(dirs.keys()) == {"RISK_INCREASING", "RISK_DECREASING"}, feat


class TestExplainPredictionShap:
    def test_runs_on_real_bundle(self):
        """Integration with the actual PAIMANA bundle: exact TreeSHAP must run."""
        import joblib
        import numpy as np

        from app.config import settings
        bundle = joblib.load(settings.MODEL_PATH)
        model = bundle["model"]
        cols = bundle["feature_columns"]

        row = np.zeros((1, len(cols)), dtype=np.float32)
        base_rate, drivers = explain_prediction_shap(model, row, cols, top_k=5)

        assert 0.0 <= base_rate <= 1.0
        assert len(drivers) == 5
        ranks = [d.rank for d in drivers]
        assert ranks == [1, 2, 3, 4, 5]
        for d in drivers:
            assert d.feature_name in cols
            assert d.direction in ("RISK_INCREASING", "RISK_DECREASING", "NEUTRAL")
            assert len(d.administrative_interpretation) > 10

    def test_top_k_respected(self):
        import joblib
        import numpy as np
        from app.config import settings

        bundle = joblib.load(settings.MODEL_PATH)
        model = bundle["model"]
        cols = bundle["feature_columns"]
        row = np.ones((1, len(cols)), dtype=np.float32)
        _, drivers = explain_prediction_shap(model, row, cols, top_k=3)
        assert len(drivers) == 3
