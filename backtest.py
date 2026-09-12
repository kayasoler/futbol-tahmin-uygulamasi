from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Callable

import pandas as pd

from analysis import build_report


ProgressCallback = Callable[[int, int], None]
STAKE = 100.0
RESULT_ODDS_COLUMNS = {"1": "b365_home", "X": "b365_draw", "2": "b365_away"}
VALUE_THRESHOLDS = (0.00, 0.03, 0.05, 0.10)
MS_PROBABILITY_THRESHOLDS = (0.45, 0.50, 0.55, 0.60, 0.62, 0.65, 0.70)
MS_MARGIN_THRESHOLDS = (0.00, 0.08, 0.12, 0.18, 0.20)
DRAW_PROBABILITY_THRESHOLDS = (0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30)
DRAW_MAX_DEFICITS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15)
DIXON_COLES_RHO_CANDIDATES = tuple(value / 100 for value in range(-15, 5))
TOTALS_25_PROBABILITY_THRESHOLDS = (
    0.50,
    0.52,
    0.55,
    0.58,
    0.60,
    0.62,
    0.65,
    0.68,
    0.70,
    0.75,
)
TOTALS_25_VALUE_THRESHOLDS = (-1.0, 0.00, 0.02, 0.04, 0.06, 0.08)


def _number(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _team_key(value: Any) -> str:
    return str(value or "").strip().casefold()


def _actual_result(row: dict[str, Any]) -> str | None:
    result = str(row.get("full_time_result") or "").strip().upper()
    if result in {"H", "D", "A"}:
        return {"H": "1", "D": "X", "A": "2"}[result]
    home_goals = _number(row.get("full_time_home_goals"))
    away_goals = _number(row.get("full_time_away_goals"))
    if home_goals is None or away_goals is None:
        return None
    return "1" if home_goals > away_goals else "2" if away_goals > home_goals else "X"


def _same_odds(row: dict[str, Any], target: dict[str, Any]) -> bool:
    for column in ("b365_home", "b365_draw", "b365_away"):
        row_value = _number(row.get(column))
        target_value = _number(target.get(column))
        if row_value is None or target_value is None or round(row_value, 2) != round(target_value, 2):
            return False
    return True


def _majority_total(rows: list[dict[str, Any]], threshold: float) -> str:
    over = under = 0
    for row in rows:
        home = _number(row.get("full_time_home_goals"))
        away = _number(row.get("full_time_away_goals"))
        if home is None or away is None:
            continue
        if home + away > threshold:
            over += 1
        else:
            under += 1
    return "Üst" if over >= under else "Alt"


def _majority_btts(rows: list[dict[str, Any]]) -> str:
    yes = no = 0
    for row in rows:
        home = _number(row.get("full_time_home_goals"))
        away = _number(row.get("full_time_away_goals"))
        if home is None or away is None:
            continue
        if home > 0 and away > 0:
            yes += 1
        else:
            no += 1
    return "KG Var" if yes >= no else "KG Yok"


def _majority_score(rows: list[dict[str, Any]]) -> str:
    scores: Counter[str] = Counter()
    for row in rows:
        home = _number(row.get("full_time_home_goals"))
        away = _number(row.get("full_time_away_goals"))
        if home is not None and away is not None:
            scores[f"{int(home)}-{int(away)}"] += 1
    return scores.most_common(1)[0][0] if scores else "—"


def _empty_bet_group() -> dict[str, float]:
    return {"bets": 0, "correct": 0, "odds_sum": 0.0, "net": 0.0}


def _record_bet(group: dict[str, float], odds: float, won: bool) -> float:
    net = STAKE * (odds - 1) if won else -STAKE
    group["bets"] += 1
    group["correct"] += int(won)
    group["odds_sum"] += odds
    group["net"] += net
    return net


def ms_confusion_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Show where 1-X-2 predictions are confused with each actual outcome."""
    counts: Counter[tuple[str, str]] = Counter()
    for record in records:
        actual = str(record.get("actual") or "")
        predicted = str(record.get("predicted") or "")
        if actual in RESULT_ODDS_COLUMNS and predicted in RESULT_ODDS_COLUMNS:
            counts[(actual, predicted)] += 1

    rows: list[dict[str, Any]] = []
    for actual in RESULT_ODDS_COLUMNS:
        total = sum(counts[(actual, predicted)] for predicted in RESULT_ODDS_COLUMNS)
        if not total:
            continue
        rows.append(
            {
                "Gerçek sonuç": actual,
                "Tahmin 1": counts[(actual, "1")],
                "Tahmin X": counts[(actual, "X")],
                "Tahmin 2": counts[(actual, "2")],
                "Toplam": total,
                "Doğru oran": counts[(actual, actual)] / total,
            }
        )
    return rows


def _wilson_lower_bound(correct: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    rate = correct / total
    denominator = 1 + z * z / total
    centre = rate + z * z / (2 * total)
    spread = z * math.sqrt((rate * (1 - rate) + z * z / (4 * total)) / total)
    return (centre - spread) / denominator


def _temporal_train_holdout(
    records: list[dict[str, Any]], train_ratio: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split each league chronologically without dividing a match date."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("division") or "Bilinmeyen")].append(record)

    training: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for division_records in grouped.values():
        ordered = sorted(division_records, key=lambda row: str(row.get("date") or ""))
        if len(ordered) < 2:
            continue
        target_split = min(
            len(ordered) - 1,
            max(1, int(len(ordered) * train_ratio)),
        )
        valid_splits = [
            index
            for index in range(1, len(ordered))
            if str(ordered[index - 1].get("date") or "")
            != str(ordered[index].get("date") or "")
        ]
        if not valid_splits:
            continue
        split = min(valid_splits, key=lambda index: abs(index - target_split))
        training.extend(ordered[:split])
        holdout.extend(ordered[split:])
    return training, holdout


def _threshold_performance(
    records: list[dict[str, Any]],
    probability_threshold: float,
    margin_threshold: float,
) -> dict[str, Any]:
    selected = [
        record
        for record in records
        if float(record.get("max_probability") or 0) >= probability_threshold
        and float(record.get("margin") or 0) >= margin_threshold
    ]
    correct = sum(bool(record.get("correct")) for record in selected)
    priced = [
        record
        for record in selected
        if (_number(record.get("odds")) or 0) > 1
    ]
    net = sum(
        STAKE * (float(record["odds"]) - 1) if record.get("correct") else -STAKE
        for record in priced
    )
    return {
        "bets": len(selected),
        "correct": correct,
        "accuracy": correct / len(selected) if selected else None,
        "coverage": len(selected) / len(records) if records else None,
        "priced_bets": len(priced),
        "roi": net / (len(priced) * STAKE) if priced else None,
    }


def ms_threshold_diagnostics(
    records: list[dict[str, Any]],
    *,
    train_ratio: float = 0.70,
    minimum_training_bets: int = 30,
) -> dict[str, Any]:
    """Choose an MS filter on older matches and report it on untouched newer matches."""
    training, holdout = _temporal_train_holdout(records, train_ratio)

    rows: list[dict[str, Any]] = []
    for probability_threshold in MS_PROBABILITY_THRESHOLDS:
        for margin_threshold in MS_MARGIN_THRESHOLDS:
            train = _threshold_performance(
                training, probability_threshold, margin_threshold
            )
            test = _threshold_performance(
                holdout, probability_threshold, margin_threshold
            )
            rows.append(
                {
                    "Olasılık eşiği": probability_threshold,
                    "Fark eşiği": margin_threshold,
                    "Eğitim seçimi": train["bets"],
                    "Eğitim doğruluğu": train["accuracy"],
                    "Yeni %30 seçim": test["bets"],
                    "Yeni %30 kapsama": test["coverage"],
                    "Yeni %30 doğruluk": test["accuracy"],
                    "Oranlı seçim": test["priced_bets"],
                    "Yeni %30 ROI": test["roi"],
                    "selected": False,
                    "_wilson": _wilson_lower_bound(train["correct"], train["bets"]),
                }
            )

    required = max(minimum_training_bets, math.ceil(len(training) * 0.05))
    eligible = [row for row in rows if int(row["Eğitim seçimi"]) >= required]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["_wilson"]),
            float(row["Eğitim doğruluğu"] or 0),
            int(row["Eğitim seçimi"]),
            -float(row["Olasılık eşiği"]),
            -float(row["Fark eşiği"]),
        ),
        default=None,
    )
    if selected is not None:
        selected["selected"] = True

    return {
        "training_count": len(training),
        "holdout_count": len(holdout),
        "minimum_training_bets": required,
        "rows": [
            {key: value for key, value in row.items() if key != "_wilson"}
            for row in rows
        ],
        "selected": (
            {key: value for key, value in selected.items() if key != "_wilson"}
            if selected is not None
            else None
        ),
    }


def _draw_rule_performance(
    records: list[dict[str, Any]],
    probability_threshold: float,
    maximum_deficit: float,
) -> dict[str, Any]:
    correct = 0
    overrides = 0
    draw_predictions = 0
    draw_correct = 0
    actual_draws = 0
    priced_bets = 0
    net = 0.0
    evaluated = 0

    for record in records:
        probabilities = record.get("probabilities") or {}
        parsed = {
            label: _number(probabilities.get(label))
            for label in RESULT_ODDS_COLUMNS
        }
        if any(value is None for value in parsed.values()):
            continue
        total = sum(float(value) for value in parsed.values())
        if total <= 0:
            continue
        normalized = {label: float(value) / total for label, value in parsed.items()}
        base_prediction = str(record.get("predicted") or "")
        actual = str(record.get("actual") or "")
        if base_prediction not in RESULT_ODDS_COLUMNS or actual not in RESULT_ODDS_COLUMNS:
            continue
        evaluated += 1
        draw_deficit = max(normalized["1"], normalized["2"]) - normalized["X"]
        prediction = base_prediction
        if (
            normalized["X"] >= probability_threshold
            and draw_deficit <= maximum_deficit
        ):
            prediction = "X"

        is_correct = prediction == actual
        correct += int(is_correct)
        overrides += int(prediction != base_prediction)
        draw_predictions += int(prediction == "X")
        draw_correct += int(prediction == "X" and actual == "X")
        actual_draws += int(actual == "X")

        odds = _number((record.get("odds_by_result") or {}).get(prediction))
        if odds is not None and odds > 1:
            priced_bets += 1
            net += STAKE * (odds - 1) if is_correct else -STAKE

    return {
        "matches": evaluated,
        "correct": correct,
        "accuracy": correct / evaluated if evaluated else None,
        "overrides": overrides,
        "draw_predictions": draw_predictions,
        "draw_correct": draw_correct,
        "draw_recall": draw_correct / actual_draws if actual_draws else None,
        "draw_precision": draw_correct / draw_predictions if draw_predictions else None,
        "priced_bets": priced_bets,
        "roi": net / (priced_bets * STAKE) if priced_bets else None,
    }


def ms_draw_rule_diagnostics(
    records: list[dict[str, Any]],
    *,
    train_ratio: float = 0.70,
    minimum_training_overrides: int = 20,
) -> dict[str, Any]:
    """Test draw overrides on old matches and score one candidate on newer matches."""
    training, holdout = _temporal_train_holdout(records, train_ratio)
    baseline_train = _draw_rule_performance(training, 2.0, -1.0)
    baseline_holdout = _draw_rule_performance(holdout, 2.0, -1.0)
    required = max(
        minimum_training_overrides,
        math.ceil(len(training) * 0.01),
    )

    rows: list[dict[str, Any]] = []
    for probability_threshold in DRAW_PROBABILITY_THRESHOLDS:
        for maximum_deficit in DRAW_MAX_DEFICITS:
            train = _draw_rule_performance(
                training, probability_threshold, maximum_deficit
            )
            test = _draw_rule_performance(
                holdout, probability_threshold, maximum_deficit
            )
            train_accuracy = train["accuracy"]
            test_accuracy = test["accuracy"]
            rows.append(
                {
                    "X olasılık eşiği": probability_threshold,
                    "X azami fark": maximum_deficit,
                    "Eğitim X müdahale": train["overrides"],
                    "Eğitim doğruluk": train_accuracy,
                    "Eğitim doğruluk farkı": (
                        train_accuracy - baseline_train["accuracy"]
                        if train_accuracy is not None
                        and baseline_train["accuracy"] is not None
                        else None
                    ),
                    "Yeni %30 X müdahale": test["overrides"],
                    "Yeni %30 doğruluk": test_accuracy,
                    "Yeni %30 doğruluk farkı": (
                        test_accuracy - baseline_holdout["accuracy"]
                        if test_accuracy is not None
                        and baseline_holdout["accuracy"] is not None
                        else None
                    ),
                    "Yeni %30 X yakalama": test["draw_recall"],
                    "Yeni %30 X kesinlik": test["draw_precision"],
                    "Yeni %30 X tahmini": test["draw_predictions"],
                    "Yeni %30 oranlı maç": test["priced_bets"],
                    "Yeni %30 ROI": test["roi"],
                    "selected": False,
                    "_train_draw_recall": train["draw_recall"] or 0.0,
                }
            )

    eligible = [
        row
        for row in rows
        if int(row["Eğitim X müdahale"]) >= required
    ]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["Eğitim doğruluk"] or 0),
            float(row["_train_draw_recall"]),
            -int(row["Eğitim X müdahale"]),
            float(row["X olasılık eşiği"]),
            -float(row["X azami fark"]),
        ),
        default=None,
    )
    if selected is not None:
        selected["selected"] = True

    public_rows = [
        {key: value for key, value in row.items() if key != "_train_draw_recall"}
        for row in rows
    ]
    public_selected = (
        {
            key: value
            for key, value in selected.items()
            if key != "_train_draw_recall"
        }
        if selected is not None
        else None
    )
    return {
        "training_count": len(training),
        "holdout_count": len(holdout),
        "minimum_training_overrides": required,
        "baseline": {
            "Eğitim doğruluk": baseline_train["accuracy"],
            "Yeni %30 doğruluk": baseline_holdout["accuracy"],
            "Yeni %30 X yakalama": baseline_holdout["draw_recall"],
            "Yeni %30 X kesinlik": baseline_holdout["draw_precision"],
            "Yeni %30 X tahmini": baseline_holdout["draw_predictions"],
            "Yeni %30 oranlı maç": baseline_holdout["priced_bets"],
            "Yeni %30 ROI": baseline_holdout["roi"],
        },
        "rows": public_rows,
        "selected": public_selected,
    }


def _totals_25_performance(
    records: list[dict[str, Any]],
    probability_threshold: float,
    value_threshold: float,
    *,
    market_reference: bool = False,
    selection_prediction_field: str = "predicted",
    selection_probability_field: str = "over_probability",
    evaluation_prediction_field: str | None = None,
    evaluation_probability_field: str | None = None,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for record in records:
        prediction = str(record.get(selection_prediction_field) or "")
        actual = str(record.get("actual") or "")
        if prediction not in {"Üst", "Alt"} or actual not in {"Üst", "Alt"}:
            continue
        over_probability = _number(record.get(selection_probability_field))
        if over_probability is None or not 0 <= over_probability <= 1:
            continue
        selected_probability = (
            over_probability if prediction == "Üst" else 1 - over_probability
        )
        if selected_probability < probability_threshold:
            continue
        odds = _number((record.get("odds") or {}).get(prediction))
        edge = (
            selected_probability - (1 / odds)
            if odds is not None and odds > 1
            else None
        )
        if value_threshold >= 0 and (edge is None or edge < value_threshold):
            continue
        selected.append(record)

    correct = 0
    profits: list[float] = []
    odds_sum = 0.0
    brier_sum = 0.0
    log_loss_sum = 0.0
    evaluated = 0
    for record in selected:
        odds_by_side = record.get("odds") or {}
        if market_reference:
            over_odds = _number(odds_by_side.get("Üst"))
            under_odds = _number(odds_by_side.get("Alt"))
            if (
                over_odds is None
                or under_odds is None
                or over_odds <= 1
                or under_odds <= 1
            ):
                continue
            inverse_total = (1 / over_odds) + (1 / under_odds)
            over_probability = (1 / over_odds) / inverse_total
            prediction = "Üst" if over_odds < under_odds else "Alt"
        else:
            prediction = str(
                record.get(evaluation_prediction_field or selection_prediction_field)
                or ""
            )
            over_probability = _number(
                record.get(
                    evaluation_probability_field or selection_probability_field
                )
            )
            if (
                prediction not in {"Üst", "Alt"}
                or over_probability is None
                or not 0 <= over_probability <= 1
            ):
                continue
        evaluated += 1
        is_correct = prediction == str(record.get("actual") or "")
        correct += int(is_correct)
        actual_over = 1.0 if str(record.get("actual") or "") == "Üst" else 0.0
        probability = min(1 - 1e-12, max(1e-12, float(over_probability)))
        brier_sum += (probability - actual_over) ** 2
        log_loss_sum -= (
            actual_over * math.log(probability)
            + (1 - actual_over) * math.log(1 - probability)
        )
        odds = _number(odds_by_side.get(prediction))
        if odds is not None and odds > 1:
            odds_sum += odds
            profits.append(odds - 1 if is_correct else -1.0)

    roi = sum(profits) / len(profits) if profits else None
    if len(profits) >= 2:
        mean = float(roi)
        variance = sum((profit - mean) ** 2 for profit in profits) / (len(profits) - 1)
        roi_lower_bound = mean - 1.96 * math.sqrt(variance / len(profits))
    else:
        roi_lower_bound = roi
    return {
        "matches": evaluated,
        "correct": correct,
        "accuracy": correct / evaluated if evaluated else None,
        "coverage": evaluated / len(records) if records else None,
        "brier": brier_sum / evaluated if evaluated else None,
        "log_loss": log_loss_sum / evaluated if evaluated else None,
        "priced_bets": len(profits),
        "average_odds": odds_sum / len(profits) if profits else None,
        "roi": roi,
        "roi_lower_bound": roi_lower_bound,
    }


def totals_25_threshold_diagnostics(
    records: list[dict[str, Any]],
    *,
    train_ratio: float = 0.70,
    minimum_training_priced_bets: int = 30,
) -> dict[str, Any]:
    """Choose a 2.5 filter on older matches and validate it on newer matches."""
    training, holdout = _temporal_train_holdout(records, train_ratio)
    baseline_training = _totals_25_performance(training, 0.0, -1.0)
    baseline = _totals_25_performance(holdout, 0.0, -1.0)
    required = max(
        minimum_training_priced_bets,
        math.ceil(int(baseline_training["priced_bets"]) * 0.05),
    )

    rows: list[dict[str, Any]] = []
    for probability_threshold in TOTALS_25_PROBABILITY_THRESHOLDS:
        for value_threshold in TOTALS_25_VALUE_THRESHOLDS:
            train = _totals_25_performance(
                training, probability_threshold, value_threshold
            )
            test = _totals_25_performance(
                holdout, probability_threshold, value_threshold
            )
            rows.append(
                {
                    "Olasılık eşiği": probability_threshold,
                    "Değer eşiği": value_threshold,
                    "Eğitim seçimi": train["matches"],
                    "Eğitim oranlı seçim": train["priced_bets"],
                    "Eğitim doğruluk": train["accuracy"],
                    "Eğitim ROI": train["roi"],
                    "Eğitim ROI alt sınırı": train["roi_lower_bound"],
                    "Yeni %30 seçim": test["matches"],
                    "Yeni %30 kapsama": test["coverage"],
                    "Yeni %30 doğruluk": test["accuracy"],
                    "Yeni %30 oranlı seçim": test["priced_bets"],
                    "Yeni %30 ortalama oran": test["average_odds"],
                    "Yeni %30 ROI": test["roi"],
                    "selected": False,
                }
            )

    eligible = [
        row
        for row in rows
        if int(row["Eğitim oranlı seçim"]) >= required
        and row["Eğitim ROI alt sınırı"] is not None
    ]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["Eğitim ROI alt sınırı"]),
            float(row["Eğitim ROI"] or 0),
            float(row["Eğitim doğruluk"] or 0),
            int(row["Eğitim oranlı seçim"]),
            -float(row["Olasılık eşiği"]),
            -float(row["Değer eşiği"]),
        ),
        default=None,
    )
    if selected is not None:
        selected["selected"] = True

    probability_threshold = (
        float(selected["Olasılık eşiği"]) if selected is not None else 2.0
    )
    value_threshold = (
        float(selected["Değer eşiği"]) if selected is not None else 2.0
    )
    candidate = _totals_25_performance(
        holdout, probability_threshold, value_threshold
    )
    market = _totals_25_performance(
        holdout,
        probability_threshold,
        value_threshold,
        market_reference=True,
    )

    league_rows: list[dict[str, Any]] = []
    for division in sorted({str(row.get("division") or "Bilinmeyen") for row in holdout}):
        league_records = [
            row
            for row in holdout
            if str(row.get("division") or "Bilinmeyen") == division
        ]
        league_candidate = _totals_25_performance(
            league_records, probability_threshold, value_threshold
        )
        league_market = _totals_25_performance(
            league_records,
            probability_threshold,
            value_threshold,
            market_reference=True,
        )
        if not league_candidate["matches"]:
            continue
        league_rows.append(
            {
                "Lig": division,
                "Seçim": league_candidate["matches"],
                "Doğruluk": league_candidate["accuracy"],
                "Oranlı seçim": league_candidate["priced_bets"],
                "Ortalama oran": league_candidate["average_odds"],
                "ROI": league_candidate["roi"],
                "Piyasa doğruluk": league_market["accuracy"],
                "Piyasa ROI": league_market["roi"],
            }
        )

    tested_leagues = len(league_rows)
    positive_roi_leagues = sum(
        row["ROI"] is not None and float(row["ROI"]) > 0 for row in league_rows
    )
    required_holdout = max(20, math.ceil(int(baseline["priced_bets"]) * 0.05))
    market_beaten = (
        market["roi"] is None
        or (
            candidate["roi"] is not None
            and float(candidate["roi"]) > float(market["roi"])
        )
    )
    passes = bool(
        selected is not None
        and int(candidate["priced_bets"]) >= required_holdout
        and candidate["roi"] is not None
        and float(candidate["roi"]) > 0
        and candidate["accuracy"] is not None
        and baseline["accuracy"] is not None
        and float(candidate["accuracy"]) > float(baseline["accuracy"])
        and market_beaten
        and tested_leagues > 0
        and positive_roi_leagues >= math.ceil(tested_leagues / 2)
    )

    return {
        "training_count": len(training),
        "holdout_count": len(holdout),
        "minimum_training_priced_bets": required,
        "minimum_holdout_priced_bets": required_holdout,
        "rows": rows,
        "selected": dict(selected) if selected is not None else None,
        "baseline": baseline,
        "candidate": candidate,
        "market": market,
        "league_rows": league_rows,
        "tested_leagues": tested_leagues,
        "positive_roi_leagues": positive_roi_leagues,
        "passes": passes,
    }


def totals_25_market_independent_diagnostics(
    records: list[dict[str, Any]],
    *,
    train_ratio: float = 0.70,
    minimum_training_priced_bets: int = 30,
) -> dict[str, Any]:
    """Compare a market-independent totals model with the live blend and market."""
    training, holdout = _temporal_train_holdout(records, train_ratio)
    shadow_fields = {
        "selection_prediction_field": "statistical_predicted",
        "selection_probability_field": "statistical_over_probability",
    }
    shadow_training = _totals_25_performance(
        training,
        0.0,
        -1.0,
        **shadow_fields,
    )
    required = max(
        minimum_training_priced_bets,
        math.ceil(int(shadow_training["priced_bets"]) * 0.05),
    )

    rows: list[dict[str, Any]] = []
    for probability_threshold in TOTALS_25_PROBABILITY_THRESHOLDS:
        for value_threshold in TOTALS_25_VALUE_THRESHOLDS:
            train = _totals_25_performance(
                training,
                probability_threshold,
                value_threshold,
                **shadow_fields,
            )
            test = _totals_25_performance(
                holdout,
                probability_threshold,
                value_threshold,
                **shadow_fields,
            )
            rows.append(
                {
                    "Olasılık eşiği": probability_threshold,
                    "Değer eşiği": value_threshold,
                    "Eğitim seçimi": train["matches"],
                    "Eğitim oranlı seçim": train["priced_bets"],
                    "Eğitim doğruluk": train["accuracy"],
                    "Eğitim ROI": train["roi"],
                    "Eğitim ROI alt sınırı": train["roi_lower_bound"],
                    "Yeni %30 seçim": test["matches"],
                    "Yeni %30 kapsama": test["coverage"],
                    "Yeni %30 doğruluk": test["accuracy"],
                    "Yeni %30 Brier": test["brier"],
                    "Yeni %30 log-loss": test["log_loss"],
                    "Yeni %30 oranlı seçim": test["priced_bets"],
                    "Yeni %30 ROI": test["roi"],
                    "selected": False,
                }
            )

    eligible = [
        row
        for row in rows
        if int(row["Eğitim oranlı seçim"]) >= required
        and row["Eğitim ROI alt sınırı"] is not None
    ]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["Eğitim ROI alt sınırı"]),
            float(row["Eğitim ROI"] or 0),
            float(row["Eğitim doğruluk"] or 0),
            int(row["Eğitim oranlı seçim"]),
            -float(row["Olasılık eşiği"]),
            -float(row["Değer eşiği"]),
        ),
        default=None,
    )
    if selected is not None:
        selected["selected"] = True

    probability_threshold = (
        float(selected["Olasılık eşiği"]) if selected is not None else 2.0
    )
    value_threshold = (
        float(selected["Değer eşiği"]) if selected is not None else 2.0
    )
    all_current = _totals_25_performance(holdout, 0.0, -1.0)
    all_shadow = _totals_25_performance(
        holdout,
        0.0,
        -1.0,
        **shadow_fields,
    )
    all_market = _totals_25_performance(
        holdout,
        0.0,
        -1.0,
        market_reference=True,
        **shadow_fields,
    )
    candidate_shadow = _totals_25_performance(
        holdout,
        probability_threshold,
        value_threshold,
        **shadow_fields,
    )
    candidate_current = _totals_25_performance(
        holdout,
        probability_threshold,
        value_threshold,
        evaluation_prediction_field="predicted",
        evaluation_probability_field="over_probability",
        **shadow_fields,
    )
    candidate_market = _totals_25_performance(
        holdout,
        probability_threshold,
        value_threshold,
        market_reference=True,
        **shadow_fields,
    )

    league_rows: list[dict[str, Any]] = []
    divisions = sorted(
        {str(record.get("division") or "Bilinmeyen") for record in holdout}
    )
    for division in divisions:
        league_records = [
            record
            for record in holdout
            if str(record.get("division") or "Bilinmeyen") == division
        ]
        league_shadow = _totals_25_performance(
            league_records,
            probability_threshold,
            value_threshold,
            **shadow_fields,
        )
        if not league_shadow["matches"]:
            continue
        league_current = _totals_25_performance(
            league_records,
            probability_threshold,
            value_threshold,
            evaluation_prediction_field="predicted",
            evaluation_probability_field="over_probability",
            **shadow_fields,
        )
        league_market = _totals_25_performance(
            league_records,
            probability_threshold,
            value_threshold,
            market_reference=True,
            **shadow_fields,
        )
        league_rows.append(
            {
                "Lig": division,
                "Seçim": league_shadow["matches"],
                "Gölge doğruluk": league_shadow["accuracy"],
                "Mevcut doğruluk": league_current["accuracy"],
                "Piyasa doğruluk": league_market["accuracy"],
                "Gölge ROI": league_shadow["roi"],
                "Mevcut ROI": league_current["roi"],
                "Piyasa ROI": league_market["roi"],
            }
        )

    tested_leagues = len(league_rows)
    positive_roi_leagues = sum(
        row["Gölge ROI"] is not None and float(row["Gölge ROI"]) > 0
        for row in league_rows
    )
    required_holdout = max(
        20,
        math.ceil(int(all_shadow["priced_bets"]) * 0.05),
    )

    def lower(candidate: dict[str, Any], reference: dict[str, Any], key: str) -> bool:
        candidate_value = candidate.get(key)
        reference_value = reference.get(key)
        return bool(
            candidate_value is not None
            and reference_value is not None
            and float(candidate_value) < float(reference_value)
        )

    def greater(candidate: dict[str, Any], reference: dict[str, Any], key: str) -> bool:
        candidate_value = candidate.get(key)
        reference_value = reference.get(key)
        return bool(
            candidate_value is not None
            and reference_value is not None
            and float(candidate_value) > float(reference_value)
        )

    passes = bool(
        selected is not None
        and int(candidate_shadow["priced_bets"]) >= required_holdout
        and greater(candidate_shadow, candidate_current, "roi")
        and greater(candidate_shadow, candidate_market, "roi")
        and greater(candidate_shadow, candidate_current, "accuracy")
        and greater(candidate_shadow, candidate_market, "accuracy")
        and lower(all_shadow, all_current, "brier")
        and lower(all_shadow, all_current, "log_loss")
        and tested_leagues > 0
        and positive_roi_leagues >= math.ceil(tested_leagues / 2)
    )

    return {
        "training_count": len(training),
        "holdout_count": len(holdout),
        "minimum_training_priced_bets": required,
        "minimum_holdout_priced_bets": required_holdout,
        "rows": rows,
        "selected": dict(selected) if selected is not None else None,
        "all_current": all_current,
        "all_shadow": all_shadow,
        "all_market": all_market,
        "candidate_current": candidate_current,
        "candidate_shadow": candidate_shadow,
        "candidate_market": candidate_market,
        "league_rows": league_rows,
        "tested_leagues": tested_leagues,
        "positive_roi_leagues": positive_roi_leagues,
        "passes": passes,
    }


def _dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    expected_home: float,
    expected_away: float,
    rho: float,
) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1 - expected_home * expected_away * rho
    if home_goals == 0 and away_goals == 1:
        return 1 + expected_home * rho
    if home_goals == 1 and away_goals == 0:
        return 1 + expected_away * rho
    if home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


def _dixon_coles_ms_probabilities(
    expected_home: float,
    expected_away: float,
    rho: float,
) -> dict[str, float] | None:
    score_grid: dict[tuple[int, int], float] = {}
    for home_goals in range(9):
        for away_goals in range(9):
            tau = _dixon_coles_tau(
                home_goals,
                away_goals,
                expected_home,
                expected_away,
                rho,
            )
            if tau <= 0:
                return None
            probability = (
                math.exp(-expected_home)
                * expected_home**home_goals
                / math.factorial(home_goals)
                * math.exp(-expected_away)
                * expected_away**away_goals
                / math.factorial(away_goals)
                * tau
            )
            score_grid[(home_goals, away_goals)] = probability

    total = sum(score_grid.values())
    if total <= 0:
        return None
    result_probabilities = {"1": 0.0, "X": 0.0, "2": 0.0}
    for (home_goals, away_goals), probability in score_grid.items():
        result = "1" if home_goals > away_goals else "2" if away_goals > home_goals else "X"
        result_probabilities[result] += probability / total
    return result_probabilities


def _normalized_probabilities(values: Any) -> dict[str, float] | None:
    if not isinstance(values, dict):
        return None
    parsed = {label: _number(values.get(label)) for label in RESULT_ODDS_COLUMNS}
    if any(value is None or float(value) < 0 for value in parsed.values()):
        return None
    total = sum(float(value) for value in parsed.values())
    if total <= 0:
        return None
    return {label: float(value) / total for label, value in parsed.items()}


def _dixon_coles_candidate_probabilities(
    record: dict[str, Any], rho: float
) -> dict[str, float] | None:
    current = _normalized_probabilities(record.get("current_probabilities"))
    components = record.get("components") or {}
    poisson = next(
        (
            component
            for name, component in components.items()
            if str(name).startswith("Poisson") and isinstance(component, dict)
        ),
        None,
    )
    if current is None or poisson is None:
        return None
    poisson_current = _normalized_probabilities(poisson.get("probabilities"))
    poisson_weight = _number(poisson.get("weight"))
    expected_home = _number(record.get("expected_home_goals"))
    expected_away = _number(record.get("expected_away_goals"))
    if (
        poisson_current is None
        or poisson_weight is None
        or not 0 < poisson_weight <= 1
        or expected_home is None
        or expected_away is None
        or expected_home <= 0
        or expected_away <= 0
    ):
        return None
    corrected = _dixon_coles_ms_probabilities(expected_home, expected_away, rho)
    if corrected is None:
        return None
    candidate = {
        label: max(
            1e-9,
            current[label]
            + poisson_weight * (corrected[label] - poisson_current[label]),
        )
        for label in RESULT_ODDS_COLUMNS
    }
    return _normalized_probabilities(candidate)


def _fit_dixon_coles_rho(
    records: list[dict[str, Any]],
) -> tuple[float, int, int]:
    usable: list[tuple[int, int, float, float]] = []
    for record in records:
        home_goals = _number(record.get("actual_home_goals"))
        away_goals = _number(record.get("actual_away_goals"))
        expected_home = _number(record.get("expected_home_goals"))
        expected_away = _number(record.get("expected_away_goals"))
        if None in (home_goals, away_goals, expected_home, expected_away):
            continue
        usable.append(
            (
                int(float(home_goals)),
                int(float(away_goals)),
                float(expected_home),
                float(expected_away),
            )
        )

    low_score_count = sum(
        home_goals <= 1 and away_goals <= 1
        for home_goals, away_goals, _, _ in usable
    )
    best_rho = 0.0
    best_key = (float("-inf"), float("-inf"))
    for rho in DIXON_COLES_RHO_CANDIDATES:
        log_likelihood = 0.0
        valid = True
        for home_goals, away_goals, expected_home, expected_away in usable:
            tau = _dixon_coles_tau(
                home_goals,
                away_goals,
                expected_home,
                expected_away,
                rho,
            )
            if tau <= 0:
                valid = False
                break
            log_likelihood += math.log(tau)
        if not valid:
            continue
        key = (log_likelihood, -abs(rho))
        if key > best_key:
            best_key = key
            best_rho = rho
    return best_rho, len(usable), low_score_count


def _multiclass_event(
    record: dict[str, Any], probabilities: dict[str, float]
) -> dict[str, Any] | None:
    actual = str(record.get("actual") or "")
    if actual not in RESULT_ODDS_COLUMNS:
        return None
    predicted = max(RESULT_ODDS_COLUMNS, key=lambda label: probabilities[label])
    correct = predicted == actual
    brier = sum(
        (probabilities[label] - (1.0 if label == actual else 0.0)) ** 2
        for label in RESULT_ODDS_COLUMNS
    ) / 3
    log_loss = -math.log(max(probabilities[actual], 1e-12))
    odds = _number((record.get("odds") or {}).get(predicted))
    profit = None
    if odds is not None and odds > 1:
        profit = odds - 1 if correct else -1.0
    return {
        "actual": actual,
        "predicted": predicted,
        "correct": correct,
        "brier": brier,
        "log_loss": log_loss,
        "profit": profit,
    }


def _summarize_multiclass_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(events)
    draw_predictions = sum(event["predicted"] == "X" for event in events)
    actual_draws = sum(event["actual"] == "X" for event in events)
    draw_correct = sum(
        event["predicted"] == "X" and event["actual"] == "X"
        for event in events
    )
    priced = [event for event in events if event["profit"] is not None]
    return {
        "matches": total,
        "accuracy": (
            sum(bool(event["correct"]) for event in events) / total
            if total
            else None
        ),
        "brier": (
            sum(float(event["brier"]) for event in events) / total
            if total
            else None
        ),
        "log_loss": (
            sum(float(event["log_loss"]) for event in events) / total
            if total
            else None
        ),
        "draw_recall": draw_correct / actual_draws if actual_draws else None,
        "draw_precision": draw_correct / draw_predictions if draw_predictions else None,
        "draw_predictions": draw_predictions,
        "priced_bets": len(priced),
        "roi": (
            sum(float(event["profit"]) for event in priced) / len(priced)
            if priced
            else None
        ),
    }


def dixon_coles_diagnostics(
    records: list[dict[str, Any]],
    *,
    train_ratio: float = 0.70,
    minimum_low_score_matches: int = 10,
) -> dict[str, Any]:
    """Fit league-level rho on old scores and compare paired MS forecasts on new scores."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("division") or "Bilinmeyen")].append(record)

    all_current_events: list[dict[str, Any]] = []
    all_candidate_events: list[dict[str, Any]] = []
    league_rows: list[dict[str, Any]] = []
    for division, division_records in sorted(grouped.items()):
        training, holdout = _temporal_train_holdout(division_records, train_ratio)
        rho, fitted_matches, low_score_matches = _fit_dixon_coles_rho(training)
        if low_score_matches < minimum_low_score_matches:
            league_rows.append(
                {
                    "Lig": division,
                    "Rho": None,
                    "Eğitim maçı": fitted_matches,
                    "Eğitim düşük skor": low_score_matches,
                    "Yeni %30 maç": 0,
                    "Mevcut doğruluk": None,
                    "DC doğruluk": None,
                    "Doğruluk farkı": None,
                    "Brier farkı": None,
                    "Log-loss farkı": None,
                    "ROI farkı": None,
                    "Durum": "Yetersiz düşük skor örneği",
                }
            )
            continue

        current_events: list[dict[str, Any]] = []
        candidate_events: list[dict[str, Any]] = []
        for record in holdout:
            current_probabilities = _normalized_probabilities(
                record.get("current_probabilities")
            )
            candidate_probabilities = _dixon_coles_candidate_probabilities(
                record, rho
            )
            if current_probabilities is None or candidate_probabilities is None:
                continue
            current_event = _multiclass_event(record, current_probabilities)
            candidate_event = _multiclass_event(record, candidate_probabilities)
            if current_event is None or candidate_event is None:
                continue
            current_events.append(current_event)
            candidate_events.append(candidate_event)

        current = _summarize_multiclass_events(current_events)
        candidate = _summarize_multiclass_events(candidate_events)
        all_current_events.extend(current_events)
        all_candidate_events.extend(candidate_events)

        def difference(key: str) -> float | None:
            current_value = current.get(key)
            candidate_value = candidate.get(key)
            if current_value is None or candidate_value is None:
                return None
            return float(candidate_value) - float(current_value)

        league_rows.append(
            {
                "Lig": division,
                "Rho": rho,
                "Eğitim maçı": fitted_matches,
                "Eğitim düşük skor": low_score_matches,
                "Yeni %30 maç": candidate["matches"],
                "Mevcut doğruluk": current["accuracy"],
                "DC doğruluk": candidate["accuracy"],
                "Doğruluk farkı": difference("accuracy"),
                "Brier farkı": difference("brier"),
                "Log-loss farkı": difference("log_loss"),
                "ROI farkı": difference("roi"),
                "Durum": "Test edildi" if candidate["matches"] else "Eşleşen kayıt yok",
            }
        )

    current = _summarize_multiclass_events(all_current_events)
    candidate = _summarize_multiclass_events(all_candidate_events)
    tested_leagues = sum(row["Durum"] == "Test edildi" for row in league_rows)
    improved_leagues = sum(
        row["Durum"] == "Test edildi"
        and row["Doğruluk farkı"] is not None
        and float(row["Doğruluk farkı"]) > 0
        for row in league_rows
    )

    def combined_difference(key: str) -> float | None:
        current_value = current.get(key)
        candidate_value = candidate.get(key)
        if current_value is None or candidate_value is None:
            return None
        return float(candidate_value) - float(current_value)

    return {
        "current": current,
        "candidate": candidate,
        "differences": {
            "accuracy": combined_difference("accuracy"),
            "brier": combined_difference("brier"),
            "log_loss": combined_difference("log_loss"),
            "roi": combined_difference("roi"),
        },
        "league_rows": league_rows,
        "tested_leagues": tested_leagues,
        "improved_leagues": improved_leagues,
        "passes": bool(
            candidate["matches"]
            and combined_difference("accuracy") is not None
            and combined_difference("accuracy") > 0
            and combined_difference("brier") is not None
            and combined_difference("brier") < 0
            and combined_difference("log_loss") is not None
            and combined_difference("log_loss") < 0
            and (
                combined_difference("roi") is None
                or combined_difference("roi") >= 0
            )
            and tested_leagues > 0
            and improved_leagues >= math.ceil(tested_leagues / 2)
        ),
    }


def _model_match(row: dict[str, Any]) -> dict[str, Any]:
    """Remove the answer fields before asking the model for a prediction."""
    hidden = {
        "full_time_home_goals",
        "full_time_away_goals",
        "full_time_result",
        "half_time_home_goals",
        "half_time_away_goals",
        "half_time_result",
    }
    return {key: value for key, value in row.items() if key not in hidden}


def run_backtest(
    league_rows: list[dict[str, Any]],
    test_size: int = 300,
    minimum_history: int = 200,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Walk forward through completed matches without exposing future results."""
    dated_rows: list[tuple[pd.Timestamp, dict[str, Any]]] = []
    for row in league_rows:
        match_date = pd.to_datetime(row.get("match_date"), errors="coerce")
        if pd.isna(match_date) or _actual_result(row) is None:
            continue
        if _number(row.get("full_time_home_goals")) is None or _number(row.get("full_time_away_goals")) is None:
            continue
        dated_rows.append((pd.Timestamp(match_date), row))
    dated_rows.sort(
        key=lambda item: (
            item[0],
            _team_key(item[1].get("home_team")),
            _team_key(item[1].get("away_team")),
        )
    )

    eligible: list[tuple[pd.Timestamp, dict[str, Any]]] = []
    for index, item in enumerate(dated_rows):
        if index >= minimum_history:
            eligible.append(item)
    targets = eligible[-max(1, int(test_size)) :]
    if not targets:
        raise ValueError(
            f"Backtest için en az {minimum_history + 1} tamamlanmış lig maçı gerekiyor."
        )

    correct = defaultdict(int)
    confidence_counts = defaultdict(int)
    confidence_correct = defaultdict(int)
    reference_correct = defaultdict(int)
    paired_model_correct = defaultdict(int)
    reference_counts = defaultdict(int)
    bet_groups = {
        "Model · tüm güvenler": _empty_bet_group(),
        "Model · Yüksek + Orta": _empty_bet_group(),
        "Model · Yüksek": _empty_bet_group(),
        "Model · Orta": _empty_bet_group(),
        "Model · Düşük": _empty_bet_group(),
        "Bet365 favorisine kör bahis": _empty_bet_group(),
    }
    value_groups = {threshold: _empty_bet_group() for threshold in VALUE_THRESHOLDS}
    details: list[dict[str, Any]] = []
    calibration_records: list[dict[str, Any]] = []
    ms_diagnostics: list[dict[str, Any]] = []
    totals_25_records: list[dict[str, Any]] = []

    for position, (target_date, target) in enumerate(targets, start=1):
        history = [row for row_date, row in dated_rows if row_date < target_date]
        if len(history) < minimum_history:
            continue

        home_key = _team_key(target.get("home_team"))
        away_key = _team_key(target.get("away_team"))
        h2h = [
            row
            for row in history
            if {_team_key(row.get("home_team")), _team_key(row.get("away_team"))}
            == {home_key, away_key}
        ][-10:]
        home_form = [
            row for row in history if _team_key(row.get("home_team")) == home_key
        ][-10:]
        away_form = [
            row for row in history if _team_key(row.get("away_team")) == away_key
        ][-10:]
        same_odds = [row for row in history if _same_odds(row, target)]

        report = build_report(
            _model_match(target),
            h2h,
            same_odds,
            history,
            home_form_rows=home_form,
            away_form_rows=away_form,
            same_odds_all_rows=same_odds,
        )
        predictions = report["predictions"]
        predicted_ms = str(predictions["ms"]).replace("MS", "").strip()
        actual_ms = _actual_result(target)
        home_goals = int(float(target["full_time_home_goals"]))
        away_goals = int(float(target["full_time_away_goals"]))
        actual_score = f"{home_goals}-{away_goals}"
        total_goals = home_goals + away_goals
        actual_btts = "KG Var" if home_goals > 0 and away_goals > 0 else "KG Yok"
        confidence = str(predictions.get("confidence") or "Bilinmiyor")

        tested = len(details) + 1
        ms_is_correct = predicted_ms == actual_ms
        correct["MS"] += int(ms_is_correct)
        correct["Skor"] += int(str(predictions["score"]) == actual_score)
        correct["KG"] += int(str(predictions["btts_prediction"]) == actual_btts)
        confidence_counts[confidence] += 1
        confidence_correct[confidence] += int(ms_is_correct)

        model_odds = _number(target.get(RESULT_ODDS_COLUMNS.get(predicted_ms, "")))
        model_net: float | None = None
        value_edge: float | None = None
        value_result: str | None = None
        if model_odds is not None and model_odds > 1:
            model_net = _record_bet(
                bet_groups["Model · tüm güvenler"], model_odds, ms_is_correct
            )
            confidence_group = f"Model · {confidence}"
            if confidence_group in bet_groups:
                _record_bet(bet_groups[confidence_group], model_odds, ms_is_correct)
            if confidence in {"Yüksek", "Orta"}:
                _record_bet(
                    bet_groups["Model · Yüksek + Orta"], model_odds, ms_is_correct
                )

        probabilities = predictions.get("ms_probabilities") or {}
        parsed_probabilities = {
            result_code: _number(probabilities.get(result_code))
            for result_code in RESULT_ODDS_COLUMNS
        }
        if all(value is not None for value in parsed_probabilities.values()):
            probability_total = sum(
                float(value) for value in parsed_probabilities.values()
            )
            if probability_total > 0:
                normalized_by_result = {
                    result_code: float(value) / probability_total
                    for result_code, value in parsed_probabilities.items()
                }
                normalized_probabilities = sorted(
                    normalized_by_result.values(), reverse=True
                )
                ms_diagnostics.append(
                    {
                        "date": str(target.get("match_date") or ""),
                        "division": str(target.get("division") or ""),
                        "predicted": predicted_ms,
                        "actual": actual_ms,
                        "max_probability": normalized_probabilities[0],
                        "margin": normalized_probabilities[0] - normalized_probabilities[1],
                        "odds": model_odds,
                        "probabilities": normalized_by_result,
                        "odds_by_result": {
                            result_code: _number(target.get(odds_column))
                            for result_code, odds_column in RESULT_ODDS_COLUMNS.items()
                        },
                        "correct": ms_is_correct,
                    }
                )
        value_candidates: list[tuple[float, str, float]] = []
        for result_code, odds_column in RESULT_ODDS_COLUMNS.items():
            probability = _number(probabilities.get(result_code))
            result_odds_value = _number(target.get(odds_column))
            if probability is None or result_odds_value is None or result_odds_value <= 1:
                continue
            value_candidates.append(
                (probability - (1 / result_odds_value), result_code, result_odds_value)
            )
        if value_candidates:
            value_edge, value_result, value_odds = max(value_candidates, key=lambda item: item[0])
            value_won = value_result == actual_ms
            for threshold in VALUE_THRESHOLDS:
                if value_edge >= threshold:
                    _record_bet(value_groups[threshold], value_odds, value_won)

        result_odds = {
            result: _number(target.get(column))
            for result, column in RESULT_ODDS_COLUMNS.items()
        }
        if all(value is not None and value > 1 for value in result_odds.values()):
            favorite = min(result_odds, key=result_odds.get)
            favorite_won = favorite == actual_ms
            reference_counts["MS"] += 1
            reference_correct["MS"] += int(favorite_won)
            paired_model_correct["MS"] += int(ms_is_correct)
            _record_bet(
                bet_groups["Bet365 favorisine kör bahis"],
                float(result_odds[favorite]),
                favorite_won,
            )

        total_results: dict[str, str] = {}
        for threshold in ("0.5", "1.5", "2.5", "3.5"):
            actual_total = "Üst" if total_goals > float(threshold) else "Alt"
            predicted_total = str(predictions["totals"][threshold]["prediction"])
            correct[threshold] += int(predicted_total == actual_total)
            reference_prediction = _majority_total(history, float(threshold))
            reference_counts[threshold] += 1
            reference_correct[threshold] += int(reference_prediction == actual_total)
            paired_model_correct[threshold] += int(predicted_total == actual_total)
            total_results[threshold] = "✓" if predicted_total == actual_total else "✗"

        total_25 = predictions["totals"]["2.5"]
        totals_25_records.append(
            {
                "date": str(target.get("match_date") or ""),
                "division": str(target.get("division") or ""),
                "predicted": str(total_25["prediction"]),
                "actual": "Üst" if total_goals > 2.5 else "Alt",
                "over_probability": _number(total_25.get("probability")),
                "statistical_predicted": str(
                    total_25.get("statistical_prediction") or ""
                ),
                "statistical_over_probability": _number(
                    total_25.get("statistical_probability")
                ),
                "odds": {
                    "Üst": _number(target.get("b365_over_25")),
                    "Alt": _number(target.get("b365_under_25")),
                },
            }
        )

        reference_counts["KG"] += 1
        reference_correct["KG"] += int(_majority_btts(history) == actual_btts)
        paired_model_correct["KG"] += int(
            str(predictions["btts_prediction"]) == actual_btts
        )
        reference_counts["Skor"] += 1
        reference_correct["Skor"] += int(_majority_score(history) == actual_score)
        paired_model_correct["Skor"] += int(str(predictions["score"]) == actual_score)

        details.append(
            {
                "Tarih": str(target.get("match_date") or ""),
                "Maç": f"{target.get('home_team')} — {target.get('away_team')}",
                "Tahmin / Gerçek MS": f"{predicted_ms} / {actual_ms}",
                "Tahmin / Gerçek skor": f"{predictions['score']} / {actual_score}",
                "Güven": confidence,
                "Model MS oranı": f"{model_odds:.2f}" if model_odds is not None else "—",
                "Değer farkı": f"{value_edge * 100:+.1f} puan" if value_edge is not None else "—",
                "Değer seçimi": value_result or "—",
                "100 birim net": f"{model_net:+.1f}" if model_net is not None else "—",
                "MS": "✓" if ms_is_correct else "✗",
                "2.5": total_results["2.5"],
                "KG": "✓" if str(predictions["btts_prediction"]) == actual_btts else "✗",
            }
        )
        calibration_records.append(
            {
                "date": str(target.get("match_date") or ""),
                "division": str(target.get("division") or ""),
                "actual": actual_ms,
                "actual_home_goals": home_goals,
                "actual_away_goals": away_goals,
                "expected_home_goals": _number(
                    predictions.get("expected_home_goals")
                ),
                "expected_away_goals": _number(
                    predictions.get("expected_away_goals")
                ),
                "odds": {
                    result: _number(target.get(column))
                    for result, column in RESULT_ODDS_COLUMNS.items()
                },
                "current_probabilities": {
                    result: float((predictions.get("ms_probabilities") or {}).get(result, 0))
                    for result in ("1", "X", "2")
                },
                "components": {
                    str(component["Kaynak"]): {
                        "probabilities": {
                            result: float(component[result]) for result in ("1", "X", "2")
                        },
                        "weight": float(component.get("Ağırlık") or 0),
                        "sample": int(component.get("Örneklem") or 0),
                    }
                    for component in report.get("components") or []
                },
            }
        )
        if progress_callback and (position == 1 or position == len(targets) or position % 5 == 0):
            progress_callback(position, len(targets))

    tested = len(details)
    if not tested:
        raise ValueError("Seçilen aralıkta test edilebilir maç bulunamadı.")

    metric_labels = [
        ("MS", "Maç sonucu"),
        ("0.5", "0.5 Alt/Üst"),
        ("1.5", "1.5 Alt/Üst"),
        ("2.5", "2.5 Alt/Üst"),
        ("3.5", "3.5 Alt/Üst"),
        ("KG", "Karşılıklı gol"),
        ("Skor", "Kesin skor"),
    ]
    metrics = [
        {
            "Ölçüm": label,
            "Doğru": correct[key],
            "Test": tested,
            "Başarı": correct[key] / tested,
        }
        for key, label in metric_labels
    ]
    confidence_metrics = [
        {
            "Güven seviyesi": level,
            "Maç": confidence_counts[level],
            "MS doğru": confidence_correct[level],
            "MS başarısı": confidence_correct[level] / confidence_counts[level],
        }
        for level in ("Yüksek", "Orta", "Düşük", "Bilinmiyor")
        if confidence_counts[level]
    ]
    comparison_labels = [
        ("MS", "Maç sonucu", "Bet365'in en düşük oranlı favorisi"),
        ("0.5", "0.5 Alt/Üst", "O tarihe kadarki lig çoğunluğu"),
        ("1.5", "1.5 Alt/Üst", "O tarihe kadarki lig çoğunluğu"),
        ("2.5", "2.5 Alt/Üst", "O tarihe kadarki lig çoğunluğu"),
        ("3.5", "3.5 Alt/Üst", "O tarihe kadarki lig çoğunluğu"),
        ("KG", "Karşılıklı gol", "O tarihe kadarki lig çoğunluğu"),
        ("Skor", "Kesin skor", "O tarihe kadarki en sık skor"),
    ]
    comparisons = []
    for key, label, method in comparison_labels:
        count = reference_counts[key]
        if not count:
            continue
        model_rate = paired_model_correct[key] / count
        reference_rate = reference_correct[key] / count
        comparisons.append(
            {
                "Pazar": label,
                "Referans": method,
                "Maç": count,
                "Model başarısı": model_rate,
                "Referans başarısı": reference_rate,
                "Model farkı": model_rate - reference_rate,
            }
        )

    profit_metrics = []
    for label in (
        "Model · tüm güvenler",
        "Model · Yüksek + Orta",
        "Model · Yüksek",
        "Model · Orta",
        "Model · Düşük",
        "Bet365 favorisine kör bahis",
    ):
        group = bet_groups[label]
        bets = int(group["bets"])
        if not bets:
            continue
        invested = bets * STAKE
        profit_metrics.append(
            {
                "Strateji": label,
                "Sanal bahis": bets,
                "Doğru": int(group["correct"]),
                "Ortalama oran": group["odds_sum"] / bets,
                "Yatırılan": invested,
                "Net sonuç": group["net"],
                "ROI": group["net"] / invested,
            }
        )

    value_metrics = []
    for threshold in VALUE_THRESHOLDS:
        group = value_groups[threshold]
        bets = int(group["bets"])
        if not bets:
            value_metrics.append(
                {
                    "Değer eşiği": f"En az +{threshold * 100:.0f} puan",
                    "Eşik": threshold,
                    "Sanal bahis": 0,
                    "Doğru": 0,
                    "Ortalama oran": None,
                    "Yatırılan": 0.0,
                    "Net sonuç": 0.0,
                    "ROI": None,
                }
            )
            continue
        invested = bets * STAKE
        value_metrics.append(
            {
                "Değer eşiği": f"En az +{threshold * 100:.0f} puan",
                "Eşik": threshold,
                "Sanal bahis": bets,
                "Doğru": int(group["correct"]),
                "Ortalama oran": group["odds_sum"] / bets,
                "Yatırılan": invested,
                "Net sonuç": group["net"],
                "ROI": group["net"] / invested,
            }
        )
    return {
        "tested": tested,
        "metrics": metrics,
        "confidence_metrics": confidence_metrics,
        "comparisons": comparisons,
        "profit_metrics": profit_metrics,
        "value_metrics": value_metrics,
        "ms_diagnostics": ms_diagnostics,
        "ms_confusion": ms_confusion_rows(ms_diagnostics),
        "ms_threshold_diagnostics": ms_threshold_diagnostics(ms_diagnostics),
        "ms_draw_rule_diagnostics": ms_draw_rule_diagnostics(ms_diagnostics),
        "dixon_coles_diagnostics": dixon_coles_diagnostics(calibration_records),
        "totals_25_diagnostics": totals_25_threshold_diagnostics(totals_25_records),
        "totals_25_market_independent_diagnostics": (
            totals_25_market_independent_diagnostics(totals_25_records)
        ),
        "totals_25_records": totals_25_records,
        "calibration_records": calibration_records,
        "details": list(reversed(details)),
        "note": (
            "Her maç yalnızca kendi tarihinden önce oynanmış aynı lig maçlarıyla tahmin edildi. "
            "Kâr testi, Bet365 CSV oranında her MS seçimine sabit 100 birim sanal bahis varsayar; "
            "gerçek para, vergi veya komisyon içermez. Diğer liglerdeki aynı oran bileşeni, "
            "ücretsiz sunucuyu yormamak için backtestte kullanılmadı."
        ),
    }


def aggregate_backtests(
    league_results: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Combine independent league backtests without mixing their match histories."""
    if not league_results:
        raise ValueError("Birleştirilecek lig testi bulunamadı.")

    combined_metrics: dict[str, dict[str, float]] = defaultdict(
        lambda: {"correct": 0.0, "test": 0.0}
    )
    combined_profit: dict[str, dict[str, float]] = defaultdict(_empty_bet_group)
    combined_value = {threshold: _empty_bet_group() for threshold in VALUE_THRESHOLDS}
    league_summary: list[dict[str, Any]] = []
    combined_ms_diagnostics: list[dict[str, Any]] = []
    combined_calibration_records: list[dict[str, Any]] = []
    combined_totals_25_records: list[dict[str, Any]] = []

    for division, result in league_results:
        combined_ms_diagnostics.extend(result.get("ms_diagnostics") or [])
        combined_calibration_records.extend(result.get("calibration_records") or [])
        combined_totals_25_records.extend(result.get("totals_25_records") or [])
        for row in result.get("metrics") or []:
            item = combined_metrics[str(row["Ölçüm"])]
            item["correct"] += float(row["Doğru"])
            item["test"] += float(row["Test"])

        profit_lookup = {
            str(row["Strateji"]): row for row in result.get("profit_metrics") or []
        }
        for strategy, row in profit_lookup.items():
            bets = int(row["Sanal bahis"])
            group = combined_profit[strategy]
            group["bets"] += bets
            group["correct"] += int(row["Doğru"])
            group["odds_sum"] += float(row["Ortalama oran"]) * bets
            group["net"] += float(row["Net sonuç"])

        value_lookup = {
            float(row["Eşik"]): row for row in result.get("value_metrics") or []
        }
        for threshold, row in value_lookup.items():
            bets = int(row["Sanal bahis"])
            if not bets:
                continue
            group = combined_value[threshold]
            group["bets"] += bets
            group["correct"] += int(row["Doğru"])
            group["odds_sum"] += float(row["Ortalama oran"]) * bets
            group["net"] += float(row["Net sonuç"])

        metric_lookup = {
            str(row["Ölçüm"]): row for row in result.get("metrics") or []
        }
        confidence_lookup = {
            str(row["Güven seviyesi"]): row
            for row in result.get("confidence_metrics") or []
        }
        value_five = value_lookup.get(0.05, {})

        def profit_roi(strategy: str) -> float | None:
            row = profit_lookup.get(strategy)
            return None if not row else float(row["ROI"])

        league_summary.append(
            {
                "Lig": division,
                "Test": int(result["tested"]),
                "MS başarısı": float(metric_lookup["Maç sonucu"]["Başarı"]),
                "Yüksek güven MS": (
                    float(confidence_lookup["Yüksek"]["MS başarısı"])
                    if "Yüksek" in confidence_lookup
                    else None
                ),
                "Tüm seçimler ROI": profit_roi("Model · tüm güvenler"),
                "Yüksek güven ROI": profit_roi("Model · Yüksek"),
                "Bet365 favorisi ROI": profit_roi("Bet365 favorisine kör bahis"),
                "+5 değer bahsi": int(value_five.get("Sanal bahis") or 0),
                "+5 değer ROI": (
                    float(value_five["ROI"])
                    if value_five.get("ROI") is not None
                    else None
                ),
            }
        )

    metric_order = [
        "Maç sonucu",
        "0.5 Alt/Üst",
        "1.5 Alt/Üst",
        "2.5 Alt/Üst",
        "3.5 Alt/Üst",
        "Karşılıklı gol",
        "Kesin skor",
    ]
    metrics = []
    for label in metric_order:
        item = combined_metrics.get(label)
        if not item or not item["test"]:
            continue
        metrics.append(
            {
                "Ölçüm": label,
                "Doğru": int(item["correct"]),
                "Test": int(item["test"]),
                "Başarı": item["correct"] / item["test"],
            }
        )

    strategy_order = [
        "Model · tüm güvenler",
        "Model · Yüksek + Orta",
        "Model · Yüksek",
        "Model · Orta",
        "Model · Düşük",
        "Bet365 favorisine kör bahis",
    ]
    profit_metrics = []
    for strategy in strategy_order:
        group = combined_profit.get(strategy)
        if not group or not group["bets"]:
            continue
        bets = int(group["bets"])
        invested = bets * STAKE
        profit_metrics.append(
            {
                "Strateji": strategy,
                "Sanal bahis": bets,
                "Doğru": int(group["correct"]),
                "Ortalama oran": group["odds_sum"] / bets,
                "Yatırılan": invested,
                "Net sonuç": group["net"],
                "ROI": group["net"] / invested,
            }
        )

    value_metrics = []
    for threshold in VALUE_THRESHOLDS:
        group = combined_value[threshold]
        bets = int(group["bets"])
        invested = bets * STAKE
        value_metrics.append(
            {
                "Değer eşiği": f"En az +{threshold * 100:.0f} puan",
                "Eşik": threshold,
                "Sanal bahis": bets,
                "Doğru": int(group["correct"]),
                "Ortalama oran": group["odds_sum"] / bets if bets else None,
                "Yatırılan": invested,
                "Net sonuç": group["net"],
                "ROI": group["net"] / invested if invested else None,
            }
        )

    return {
        "tested": sum(int(result["tested"]) for _, result in league_results),
        "league_count": len(league_results),
        "metrics": metrics,
        "profit_metrics": profit_metrics,
        "value_metrics": value_metrics,
        "ms_diagnostics": combined_ms_diagnostics,
        "ms_confusion": ms_confusion_rows(combined_ms_diagnostics),
        "ms_threshold_diagnostics": ms_threshold_diagnostics(
            combined_ms_diagnostics
        ),
        "ms_draw_rule_diagnostics": ms_draw_rule_diagnostics(
            combined_ms_diagnostics
        ),
        "dixon_coles_diagnostics": dixon_coles_diagnostics(
            combined_calibration_records
        ),
        "totals_25_diagnostics": totals_25_threshold_diagnostics(
            combined_totals_25_records
        ),
        "totals_25_market_independent_diagnostics": (
            totals_25_market_independent_diagnostics(combined_totals_25_records)
        ),
        "league_summary": league_summary,
        "calibration_leagues": [
            {
                "division": division,
                "records": result.get("calibration_records") or [],
            }
            for division, result in league_results
        ],
        "note": (
            "Her lig kendi geçmişi içinde ayrı ayrı test edildi; liglerin geçmiş verileri birbirine karıştırılmadı. "
            "Değer farkı, model olasılığı eksi seçilen Bet365 oranının başabaş olasılığıdır."
        ),
    }
