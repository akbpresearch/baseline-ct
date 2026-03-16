import csv
import json
import logging
import os

from aggregator import AggregatedMetric
from models import ExtractionResult

logger = logging.getLogger(__name__)


def export_summary_csv(aggregated: list[AggregatedMetric], path: str):
    """Export summary CSV with 8 rows (4 metrics x 2 perspectives)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    fieldnames = [
        "Metric", "Perspective",
        "Strongly Disagree %", "Disagree %", "Neutral %", "Agree %", "Strongly Agree %",
        "Top 2 Box %", "Bottom 2 Box %", "Mean", "N", "Avg Confidence",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for am in aggregated:
            d = am.distribution
            writer.writerow({
                "Metric": am.metric_name.replace("_", " ").title(),
                "Perspective": am.perspective,
                "Strongly Disagree %": d[0],
                "Disagree %": d[1],
                "Neutral %": d[2],
                "Agree %": d[3],
                "Strongly Agree %": d[4],
                "Top 2 Box %": am.top2_box,
                "Bottom 2 Box %": am.bottom2_box,
                "Mean": am.mean,
                "N": am.n,
                "Avg Confidence": am.avg_confidence,
            })

    logger.info(f"Summary CSV written to {path}")


def export_raw_csv(results: list[ExtractionResult], path: str):
    """Export raw CSV with one row per content item and all individual scores."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    fieldnames = [
        "URL", "Source Type", "Title", "Word Count", "Is Relevant", "Overall Sentiment",
    ]
    metric_names = ["purchase_intent", "uniqueness", "believability", "value_perception"]
    for mn in metric_names:
        label = mn.replace("_", " ").title()
        for perspective in ["Author", "Aggregate"]:
            fieldnames.extend([
                f"{label} {perspective} Score",
                f"{label} {perspective} Confidence",
                f"{label} {perspective} Reasoning",
            ])

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "URL": result.content_item.url,
                "Source Type": result.content_item.source_type,
                "Title": result.content_item.title,
                "Word Count": result.content_item.word_count,
                "Is Relevant": result.is_relevant,
                "Overall Sentiment": result.overall_sentiment,
            }
            for m in result.metrics:
                label = m.metric_name.replace("_", " ").title()
                row[f"{label} Author Score"] = m.author_score
                row[f"{label} Author Confidence"] = m.author_confidence
                row[f"{label} Author Reasoning"] = m.author_reasoning
                row[f"{label} Aggregate Score"] = m.aggregate_score
                row[f"{label} Aggregate Confidence"] = m.aggregate_confidence
                row[f"{label} Aggregate Reasoning"] = m.aggregate_reasoning
            writer.writerow(row)

    logger.info(f"Raw CSV written to {path}")


def export_final_json(aggregated: list[AggregatedMetric], results: list[ExtractionResult],
                      product_name: str, path: str):
    """Export final JSON with scores and confidence scores."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    total = len(results)
    relevant = sum(1 for r in results if r.is_relevant)

    # Build metrics summary
    metrics = {}
    for am in aggregated:
        key = am.metric_name
        if key not in metrics:
            metrics[key] = {}
        metrics[key][am.perspective] = {
            "mean": am.mean,
            "confidence": am.avg_confidence,
            "n": am.n,
            "top_2_box_pct": am.top2_box,
            "bottom_2_box_pct": am.bottom2_box,
            "distribution": {
                "strongly_disagree_pct": am.distribution[0],
                "disagree_pct": am.distribution[1],
                "neutral_pct": am.distribution[2],
                "agree_pct": am.distribution[3],
                "strongly_agree_pct": am.distribution[4],
            },
        }

    # Build per-item raw scores
    raw_items = []
    for r in results:
        item_scores = {}
        for m in r.metrics:
            item_scores[m.metric_name] = {
                "author": {
                    "score": m.author_score,
                    "confidence": m.author_confidence,
                    "reasoning": m.author_reasoning,
                },
                "aggregate": {
                    "score": m.aggregate_score,
                    "confidence": m.aggregate_confidence,
                    "reasoning": m.aggregate_reasoning,
                },
            }
        raw_items.append({
            "url": r.content_item.url,
            "source_type": r.content_item.source_type,
            "title": r.content_item.title,
            "word_count": r.content_item.word_count,
            "is_relevant": r.is_relevant,
            "overall_sentiment": r.overall_sentiment,
            "scores": item_scores,
        })

    output = {
        "product_name": product_name,
        "summary": {
            "total_items": total,
            "relevant_items": relevant,
            "metrics": metrics,
        },
        "items": raw_items,
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Final JSON written to {path}")
