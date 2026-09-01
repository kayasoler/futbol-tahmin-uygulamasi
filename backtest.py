from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable

import pandas as pd

from analysis import build_report


ProgressCallback = Callable[[int, int], None]
STAKE = 100.0
RESULT_ODDS_COLUMNS = {"1": "b365_home", "X": "b365_draw", "2": "b365_away"}
VALUE_THRESHOLDS = (0.00, 0.03, 0.05, 0.10)


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
            selected_probability = _number(
                (predictions.get("ms_probabilities") or {}).get(predicted_ms)
            )
            if selected_probability is not None:
                value_edge = selected_probability - (1 / model_odds)
                for threshold in VALUE_THRESHOLDS:
                    if value_edge >= threshold:
                        _record_bet(value_groups[threshold], model_odds, ms_is_correct)

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
                "100 birim net": f"{model_net:+.1f}" if model_net is not None else "—",
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

    for division, result in league_results:
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
        "league_summary": league_summary,
        "note": (
            "Her lig kendi geçmişi içinde ayrı ayrı test edildi; liglerin geçmiş verileri birbirine karıştırılmadı. "
            "Değer farkı, model olasılığı eksi seçilen Bet365 oranının başabaş olasılığıdır."
        ),
    }
