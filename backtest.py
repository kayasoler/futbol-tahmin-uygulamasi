from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import pandas as pd

from analysis import build_report


ProgressCallback = Callable[[int, int], None]


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
    dated_rows.sort(key=lambda item: item[0])

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
    details: list[dict[str, Any]] = []

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

        total_results: dict[str, str] = {}
        for threshold in ("0.5", "1.5", "2.5", "3.5"):
            actual_total = "Üst" if total_goals > float(threshold) else "Alt"
            predicted_total = str(predictions["totals"][threshold]["prediction"])
            correct[threshold] += int(predicted_total == actual_total)
            total_results[threshold] = "✓" if predicted_total == actual_total else "✗"

        details.append(
            {
                "Tarih": str(target.get("match_date") or ""),
                "Maç": f"{target.get('home_team')} — {target.get('away_team')}",
                "Tahmin / Gerçek MS": f"{predicted_ms} / {actual_ms}",
                "Tahmin / Gerçek skor": f"{predictions['score']} / {actual_score}",
                "Güven": confidence,
                "MS": "✓" if ms_is_correct else "✗",
                "2.5": total_results["2.5"],
                "KG": "✓" if str(predictions["btts_prediction"]) == actual_btts else "✗",
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
    return {
        "tested": tested,
        "metrics": metrics,
        "confidence_metrics": confidence_metrics,
        "details": list(reversed(details)),
        "note": (
            "Her maç yalnızca kendi tarihinden önce oynanmış aynı lig maçlarıyla tahmin edildi. "
            "Diğer liglerdeki aynı oran bileşeni, ücretsiz sunucuyu yormamak için backtestte kullanılmadı."
        ),
    }
