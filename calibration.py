from __future__ import annotations

from typing import Any

import numpy as np


LABELS = ("1", "X", "2")
COMPONENTS = (
    "Poisson + son saha formu",
    "Bet365 piyasa olasılığı",
    "H2H son karşılaşmalar",
    "Aynı ligde aynı oran",
)
CURRENT_WEIGHTS = np.array([0.45, 0.35, 0.10, 0.07], dtype=float)
CURRENT_WEIGHTS /= CURRENT_WEIGHTS.sum()
TEMPERATURES = (0.8, 1.0, 1.2, 1.4, 1.6, 1.8)
VALUE_THRESHOLDS = (0.00, 0.03, 0.05, 0.10)
STAKE = 100.0
MIN_RELATIVE_BRIER_IMPROVEMENT = 0.01
MIN_LOG_LOSS_IMPROVEMENT = 0.005
MIN_VALUE_BETS = 30
BOOTSTRAP_SAMPLES = 2000
MIN_LEAGUE_IMPROVEMENT_SHARE = 2 / 3
MIN_ROLLING_IMPROVEMENT_SHARE = 2 / 3


def _component_strength(name: str, sample: int) -> float:
    if name in {COMPONENTS[0], COMPONENTS[1]}:
        return 1.0
    if name == COMPONENTS[2]:
        return min(1.0, sample / 8)
    if name == COMPONENTS[3]:
        return min(1.0, sample / 12)
    return 0.0


def _normalize(values: list[float] | np.ndarray) -> np.ndarray | None:
    array = np.asarray(values, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)) or array.sum() <= 0:
        return None
    return array / array.sum()


def _prepare_records(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    component_rows = []
    strength_rows = []
    current_rows = []
    market_rows = []
    actual_rows = []
    odds_rows = []

    for record in records:
        actual = str(record.get("actual") or "")
        if actual not in LABELS:
            continue
        odds_map = record.get("odds") or {}
        try:
            odds = np.array([float(odds_map[label]) for label in LABELS], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        if not np.all(np.isfinite(odds)) or np.any(odds <= 1):
            continue
        inverse = 1 / odds
        market = inverse / inverse.sum()
        current_map = record.get("current_probabilities") or {}
        current = _normalize([current_map.get(label, 0) for label in LABELS])
        if current is None:
            continue

        components = record.get("components") or {}
        probabilities = np.zeros((len(COMPONENTS), 3), dtype=float)
        strengths = np.zeros(len(COMPONENTS), dtype=float)
        for index, name in enumerate(COMPONENTS):
            item = components.get(name)
            if not isinstance(item, dict):
                continue
            probability_map = item.get("probabilities") or {}
            normalized = _normalize([probability_map.get(label, 0) for label in LABELS])
            if normalized is None:
                continue
            probabilities[index] = normalized
            strengths[index] = _component_strength(name, int(item.get("sample") or 0))
        if strengths[0] <= 0 or strengths[1] <= 0:
            continue

        component_rows.append(probabilities)
        strength_rows.append(strengths)
        current_rows.append(current)
        market_rows.append(market)
        actual_rows.append(LABELS.index(actual))
        odds_rows.append(odds)

    if not actual_rows:
        raise ValueError("Kalibrasyon için geçerli Bet365 oranlı maç bulunamadı.")
    return {
        "components": np.asarray(component_rows, dtype=float),
        "strengths": np.asarray(strength_rows, dtype=float),
        "current": np.asarray(current_rows, dtype=float),
        "market": np.asarray(market_rows, dtype=float),
        "actual": np.asarray(actual_rows, dtype=int),
        "odds": np.asarray(odds_rows, dtype=float),
    }


def _candidate_weights() -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    units = 20
    for poisson in range(2, 15):       # %10 - %70
        for market in range(4, 17):    # %20 - %80
            for h2h in range(0, 5):    # %0 - %20
                same_odds = units - poisson - market - h2h
                if 0 <= same_odds <= 4:
                    candidates.append(
                        np.array([poisson, market, h2h, same_odds], dtype=float) / units
                    )
    return candidates


def _blend(
    components: np.ndarray,
    strengths: np.ndarray,
    weights: np.ndarray,
    temperature: float,
) -> np.ndarray:
    effective = strengths * weights[np.newaxis, :]
    denominator = np.clip(effective.sum(axis=1), 1e-12, None)
    probabilities = (components * effective[:, :, np.newaxis]).sum(axis=1)
    probabilities = probabilities / denominator[:, np.newaxis]
    probabilities = np.clip(probabilities, 1e-9, 1)
    adjusted = np.power(probabilities, 1 / temperature)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def _brier(probabilities: np.ndarray, actual: np.ndarray) -> float:
    targets = np.eye(3, dtype=float)[actual]
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def _log_loss(probabilities: np.ndarray, actual: np.ndarray) -> float:
    chosen = np.clip(probabilities[np.arange(len(actual)), actual], 1e-12, 1)
    return float(-np.mean(np.log(chosen)))


def _bootstrap_improvement_probability(
    current_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    actual: np.ndarray,
) -> float:
    """Estimate how often calibration improves mean Brier loss on resampled matches."""
    targets = np.eye(3, dtype=float)[actual]
    current_loss = np.sum((current_probabilities - targets) ** 2, axis=1)
    calibrated_loss = np.sum((calibrated_probabilities - targets) ** 2, axis=1)
    improvement = current_loss - calibrated_loss
    if not len(improvement):
        return 0.0
    rng = np.random.default_rng(20260901)
    indices = rng.integers(0, len(improvement), size=(BOOTSTRAP_SAMPLES, len(improvement)))
    return float((improvement[indices].mean(axis=1) > 0).mean())


def _fit_weights(training: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    best_score = float("inf")
    best_weights = CURRENT_WEIGHTS.copy()
    best_temperature = 1.0
    for weights in _candidate_weights():
        for temperature in TEMPERATURES:
            probabilities = _blend(
                training["components"], training["strengths"], weights, temperature
            )
            regularization = 0.0005 * float(np.sum((weights - CURRENT_WEIGHTS) ** 2))
            score = _brier(probabilities, training["actual"]) + regularization
            if score < best_score:
                best_score = score
                best_weights = weights.copy()
                best_temperature = temperature
    return best_weights, best_temperature


def _rolling_validation(
    calibration_leagues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run three non-overlapping future tests with expanding training histories."""
    windows = ((0.40, 0.60), (0.60, 0.80), (0.80, 1.00))
    rows: list[dict[str, Any]] = []
    for window_number, (train_end_ratio, test_end_ratio) in enumerate(windows, start=1):
        training_records: list[dict[str, Any]] = []
        test_records: list[dict[str, Any]] = []
        for league in calibration_leagues:
            records = sorted(
                list(league.get("records") or []),
                key=lambda record: str(record.get("date") or ""),
            )
            if len(records) < 30:
                continue
            train_end = max(1, int(len(records) * train_end_ratio))
            test_end = min(len(records), max(train_end + 1, int(len(records) * test_end_ratio)))
            training_records.extend(records[:train_end])
            test_records.extend(records[train_end:test_end])
        if not training_records or not test_records:
            continue
        try:
            training = _prepare_records(training_records)
            test = _prepare_records(test_records)
        except ValueError:
            continue
        weights, temperature = _fit_weights(training)
        calibrated_probabilities = _blend(
            test["components"], test["strengths"], weights, temperature
        )
        current = _evaluate("Mevcut model", test["current"], test["actual"], test["odds"])
        calibrated = _evaluate(
            "Kalibre model", calibrated_probabilities, test["actual"], test["odds"]
        )
        value_rows = _value_rows(
            "Kalibre model", calibrated_probabilities, test["actual"], test["odds"]
        )
        plus_three = next(row for row in value_rows if row["Değer eşiği"] == "+3 puan")
        rows.append(
            {
                "Dönem": f"İleri sınav {window_number}",
                "Eğitim oranı": train_end_ratio,
                "Sınav aralığı": f"%{train_end_ratio * 100:.0f}–%{test_end_ratio * 100:.0f}",
                "Sınav maçı": len(test["actual"]),
                "Mevcut Brier": current["Brier"],
                "Kalibre Brier": calibrated["Brier"],
                "Brier iyileşmesi": current["Brier"] - calibrated["Brier"],
                "Doğruluk farkı": calibrated["Doğruluk"] - current["Doğruluk"],
                "Kalibre ROI": calibrated["ROI"],
                "+3 değer bahsi": int(plus_three["Sanal bahis"]),
                "+3 değer ROI": plus_three["ROI"],
                "İyileşti": calibrated["Brier"] < current["Brier"],
            }
        )
    return rows


def _evaluate(
    name: str,
    probabilities: np.ndarray,
    actual: np.ndarray,
    odds: np.ndarray,
) -> dict[str, Any]:
    predicted = probabilities.argmax(axis=1)
    won = predicted == actual
    selected_odds = odds[np.arange(len(actual)), predicted]
    net = np.where(won, STAKE * (selected_odds - 1), -STAKE)
    return {
        "Yöntem": name,
        "Maç": len(actual),
        "Doğru": int(won.sum()),
        "Doğruluk": float(won.mean()),
        "Brier": _brier(probabilities, actual),
        "Log loss": _log_loss(probabilities, actual),
        "Ortalama güven": float(probabilities.max(axis=1).mean()),
        "Ortalama oran": float(selected_odds.mean()),
        "Net sonuç": float(net.sum()),
        "ROI": float(net.sum() / (len(actual) * STAKE)),
    }


def _value_rows(
    name: str,
    probabilities: np.ndarray,
    actual: np.ndarray,
    odds: np.ndarray,
) -> list[dict[str, Any]]:
    # A value selection is not necessarily the most likely result. Compare all
    # 1-X-2 outcomes with their own break-even probability and take the largest edge.
    edges = probabilities - (1 / odds)
    selected_results = edges.argmax(axis=1)
    selected_probability = probabilities[np.arange(len(actual)), selected_results]
    selected_odds = odds[np.arange(len(actual)), selected_results]
    edge = selected_probability - (1 / selected_odds)
    rows = []
    for threshold in VALUE_THRESHOLDS:
        selected = edge >= threshold
        bets = int(selected.sum())
        if not bets:
            rows.append(
                {
                    "Yöntem": name,
                    "Değer eşiği": f"+{threshold * 100:.0f} puan",
                    "Sanal bahis": 0,
                    "Doğru": 0,
                    "Ortalama oran": None,
                    "Net sonuç": 0.0,
                    "ROI": None,
                }
            )
            continue
        won = selected_results[selected] == actual[selected]
        chosen_odds = selected_odds[selected]
        net = np.where(won, STAKE * (chosen_odds - 1), -STAKE)
        rows.append(
            {
                "Yöntem": name,
                "Değer eşiği": f"+{threshold * 100:.0f} puan",
                "Sanal bahis": bets,
                "Doğru": int(won.sum()),
                "Ortalama oran": float(chosen_odds.mean()),
                "Net sonuç": float(net.sum()),
                "ROI": float(net.sum() / (bets * STAKE)),
            }
        )
    return rows


def _calibration_bins(
    probabilities: np.ndarray,
    actual: np.ndarray,
) -> list[dict[str, Any]]:
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == actual
    edges = (0.33, 0.45, 0.55, 0.65, 0.75, 1.01)
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence >= lower) & (confidence < upper)
        count = int(selected.sum())
        if count:
            rows.append(
                {
                    "Tahmin aralığı": f"%{lower * 100:.0f}–%{min(100, upper * 100):.0f}",
                    "Maç": count,
                    "Ortalama tahmin": float(confidence[selected].mean()),
                    "Gerçek başarı": float(correct[selected].mean()),
                    "Kalibrasyon farkı": float(
                        confidence[selected].mean() - correct[selected].mean()
                    ),
                }
            )
    return rows


def calibrate_model(
    calibration_leagues: list[dict[str, Any]],
    train_ratio: float = 0.70,
) -> dict[str, Any]:
    """Learn only on older records and evaluate once on untouched recent records."""
    training_records: list[dict[str, Any]] = []
    holdout_records: list[dict[str, Any]] = []
    split_summary: list[dict[str, Any]] = []
    for league in calibration_leagues:
        division = str(league.get("division") or "")
        records = sorted(
            list(league.get("records") or []),
            key=lambda record: str(record.get("date") or ""),
        )
        if len(records) < 30:
            continue
        split = min(len(records) - 1, max(1, int(len(records) * train_ratio)))
        training = records[:split]
        holdout = records[split:]
        training_records.extend(training)
        holdout_records.extend(holdout)
        split_summary.append(
            {
                "Lig": division,
                "Eski %70 eğitim": len(training),
                "Yeni %30 sınav": len(holdout),
            }
        )
    if not training_records or not holdout_records:
        raise ValueError("%70/%30 kalibrasyon bölümü oluşturulamadı.")

    training = _prepare_records(training_records)
    holdout = _prepare_records(holdout_records)
    best_weights, best_temperature = _fit_weights(training)

    trained_probabilities = _blend(
        training["components"], training["strengths"], best_weights, best_temperature
    )
    holdout_probabilities = _blend(
        holdout["components"], holdout["strengths"], best_weights, best_temperature
    )
    training_comparison = [
        _evaluate("Mevcut model", training["current"], training["actual"], training["odds"]),
        _evaluate("Kalibre model", trained_probabilities, training["actual"], training["odds"]),
    ]
    holdout_comparison = [
        _evaluate("Mevcut model", holdout["current"], holdout["actual"], holdout["odds"]),
        _evaluate("Kalibre model", holdout_probabilities, holdout["actual"], holdout["odds"]),
        _evaluate("Bet365 olasılıkları", holdout["market"], holdout["actual"], holdout["odds"]),
    ]
    comparison_lookup = {row["Yöntem"]: row for row in holdout_comparison}
    current = comparison_lookup["Mevcut model"]
    calibrated = comparison_lookup["Kalibre model"]
    market = comparison_lookup["Bet365 olasılıkları"]
    relative_brier_improvement = (current["Brier"] - calibrated["Brier"]) / current["Brier"]
    log_loss_improvement = current["Log loss"] - calibrated["Log loss"]
    bootstrap_probability = _bootstrap_improvement_probability(
        holdout["current"], holdout_probabilities, holdout["actual"]
    )

    weights_table = [
        {"Bileşen": name, "Öğrenilen ağırlık": float(best_weights[index])}
        for index, name in enumerate(COMPONENTS)
    ]
    value_comparison = _value_rows(
        "Mevcut model", holdout["current"], holdout["actual"], holdout["odds"]
    ) + _value_rows(
        "Kalibre model", holdout_probabilities, holdout["actual"], holdout["odds"]
    )
    calibrated_value_bets = sum(
        int(row["Sanal bahis"])
        for row in value_comparison
        if row["Yöntem"] == "Kalibre model" and row["Değer eşiği"] == "+3 puan"
    )

    league_diagnostics: list[dict[str, Any]] = []
    for league in calibration_leagues:
        division = str(league.get("division") or "")
        records = sorted(
            list(league.get("records") or []),
            key=lambda record: str(record.get("date") or ""),
        )
        if len(records) < 30:
            continue
        split = min(len(records) - 1, max(1, int(len(records) * train_ratio)))
        try:
            league_holdout = _prepare_records(records[split:])
        except ValueError:
            continue
        league_calibrated_probabilities = _blend(
            league_holdout["components"],
            league_holdout["strengths"],
            best_weights,
            best_temperature,
        )
        league_current = _evaluate(
            "Mevcut model",
            league_holdout["current"],
            league_holdout["actual"],
            league_holdout["odds"],
        )
        league_calibrated = _evaluate(
            "Kalibre model",
            league_calibrated_probabilities,
            league_holdout["actual"],
            league_holdout["odds"],
        )
        league_value_rows = _value_rows(
            "Kalibre model",
            league_calibrated_probabilities,
            league_holdout["actual"],
            league_holdout["odds"],
        )
        plus_three = next(
            row for row in league_value_rows if row["Değer eşiği"] == "+3 puan"
        )
        league_diagnostics.append(
            {
                "Lig": division,
                "Sınav maçı": len(league_holdout["actual"]),
                "Mevcut Brier": league_current["Brier"],
                "Kalibre Brier": league_calibrated["Brier"],
                "Brier iyileşmesi": league_current["Brier"] - league_calibrated["Brier"],
                "Doğruluk farkı": league_calibrated["Doğruluk"] - league_current["Doğruluk"],
                "Kalibre ROI": league_calibrated["ROI"],
                "+3 değer bahsi": int(plus_three["Sanal bahis"]),
                "+3 değer ROI": plus_three["ROI"],
                "İyileşti": league_calibrated["Brier"] < league_current["Brier"],
            }
        )
    improved_leagues = sum(bool(row["İyileşti"]) for row in league_diagnostics)
    league_improvement_share = (
        improved_leagues / len(league_diagnostics) if league_diagnostics else 0.0
    )
    rolling_diagnostics = _rolling_validation(calibration_leagues)
    improved_windows = sum(bool(row["İyileşti"]) for row in rolling_diagnostics)
    rolling_improvement_share = (
        improved_windows / len(rolling_diagnostics) if rolling_diagnostics else 0.0
    )
    checks = [
        {
            "Kontrol": "Göreli Brier iyileşmesi",
            "Sonuç": relative_brier_improvement,
            "Eşik": MIN_RELATIVE_BRIER_IMPROVEMENT,
            "Geçti": relative_brier_improvement >= MIN_RELATIVE_BRIER_IMPROVEMENT,
        },
        {
            "Kontrol": "Log loss iyileşmesi",
            "Sonuç": log_loss_improvement,
            "Eşik": MIN_LOG_LOSS_IMPROVEMENT,
            "Geçti": log_loss_improvement >= MIN_LOG_LOSS_IMPROVEMENT,
        },
        {
            "Kontrol": "Bootstrap güveni",
            "Sonuç": bootstrap_probability,
            "Eşik": 0.95,
            "Geçti": bootstrap_probability >= 0.95,
        },
        {
            "Kontrol": "Bet365 Brier seviyesi",
            "Sonuç": calibrated["Brier"],
            "Eşik": market["Brier"],
            "Geçti": calibrated["Brier"] <= market["Brier"],
        },
        {
            "Kontrol": "+3 değer bahsi örneklemi",
            "Sonuç": calibrated_value_bets,
            "Eşik": MIN_VALUE_BETS,
            "Geçti": calibrated_value_bets >= MIN_VALUE_BETS,
        },
        {
            "Kontrol": "Ligler arası tutarlılık",
            "Sonuç": league_improvement_share,
            "Eşik": MIN_LEAGUE_IMPROVEMENT_SHARE,
            "Geçti": league_improvement_share >= MIN_LEAGUE_IMPROVEMENT_SHARE,
        },
        {
            "Kontrol": "Dönemler arası tutarlılık",
            "Sonuç": rolling_improvement_share,
            "Eşik": MIN_ROLLING_IMPROVEMENT_SHARE,
            "Geçti": rolling_improvement_share >= MIN_ROLLING_IMPROVEMENT_SHARE,
        },
    ]
    recommended = all(bool(check["Geçti"]) for check in checks)
    boundary_weights = [
        COMPONENTS[index]
        for index, weight in enumerate(best_weights)
        if weight in {0.0, 0.8}
    ]
    return {
        "training_count": len(training["actual"]),
        "holdout_count": len(holdout["actual"]),
        "split_summary": split_summary,
        "weights": weights_table,
        "temperature": best_temperature,
        "training_comparison": training_comparison,
        "holdout_comparison": holdout_comparison,
        "value_comparison": value_comparison,
        "calibration_bins": _calibration_bins(holdout_probabilities, holdout["actual"]),
        "robustness_checks": checks,
        "league_diagnostics": league_diagnostics,
        "rolling_diagnostics": rolling_diagnostics,
        "boundary_weights": boundary_weights,
        "recommended": recommended,
        "decision": (
            "Kalibre ağırlıklar tüm sağlamlık kontrollerini geçti; yine de canlıya almadan önce yeni bir dönem testinde doğrulanmalı."
            if recommended
            else "Kalibre ağırlıklar bazı ölçümlerde iyileşse de istatistiksel sağlamlık kontrollerinin tamamını geçemedi; canlı modele uygulanmamalı."
        ),
    }
