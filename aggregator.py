import logging
from dataclasses import dataclass

from models import ExtractionResult

logger = logging.getLogger(__name__)

METRIC_NAMES = ["purchase_intent", "uniqueness", "believability", "value_perception"]
PERSPECTIVES = ["author", "aggregate"]
LIKERT_LABELS = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]


@dataclass
class AggregatedMetric:
    metric_name: str
    perspective: str  # "author" or "aggregate"
    distribution: list[float]  # 5-element list of percentages (1-5)
    top2_box: float  # % scoring 4 or 5
    bottom2_box: float  # % scoring 1 or 2
    mean: float
    n: int
    avg_confidence: float


def aggregate_results(results: list[ExtractionResult],
                      confidence_threshold: float = 0.0) -> list[AggregatedMetric]:
    """Compute confidence-weighted Likert distributions for all metrics and perspectives."""
    relevant = [r for r in results if r.is_relevant]
    logger.info(f"Aggregating {len(relevant)} relevant items (of {len(results)} total)")

    if not relevant:
        logger.warning("No relevant content found — returning empty aggregation")
        return []

    aggregated: list[AggregatedMetric] = []

    for metric_name in METRIC_NAMES:
        for perspective in PERSPECTIVES:
            # Collect (score, confidence) pairs
            pairs = []
            for result in relevant:
                for m in result.metrics:
                    if m.metric_name == metric_name:
                        score = getattr(m, f"{perspective}_score")
                        confidence = getattr(m, f"{perspective}_confidence")
                        if confidence > confidence_threshold:
                            pairs.append((score, confidence))

            if not pairs:
                aggregated.append(AggregatedMetric(
                    metric_name=metric_name,
                    perspective=perspective,
                    distribution=[0.0] * 5,
                    top2_box=0.0,
                    bottom2_box=0.0,
                    mean=0.0,
                    n=0,
                    avg_confidence=0.0,
                ))
                continue

            # Confidence-weighted distribution
            bins = [0.0] * 5  # indices 0-4 map to scores 1-5
            total_weight = 0.0
            total_confidence = 0.0

            for score, confidence in pairs:
                bins[score - 1] += confidence
                total_weight += confidence
                total_confidence += confidence

            # Normalize to percentages
            if total_weight > 0:
                distribution = [(b / total_weight) * 100 for b in bins]
            else:
                distribution = [0.0] * 5

            # Top 2 Box (scores 4+5), Bottom 2 Box (scores 1+2)
            top2 = distribution[3] + distribution[4]
            bottom2 = distribution[0] + distribution[1]

            # Weighted mean
            weighted_sum = sum(bins[i] * (i + 1) for i in range(5))
            mean = weighted_sum / total_weight if total_weight > 0 else 0.0

            avg_conf = total_confidence / len(pairs) if pairs else 0.0

            aggregated.append(AggregatedMetric(
                metric_name=metric_name,
                perspective=perspective,
                distribution=[round(d, 1) for d in distribution],
                top2_box=round(top2, 1),
                bottom2_box=round(bottom2, 1),
                mean=round(mean, 2),
                n=len(pairs),
                avg_confidence=round(avg_conf, 2),
            ))

    return aggregated


def format_summary_table(aggregated: list[AggregatedMetric]) -> str:
    """Format aggregated results as a pretty terminal table."""
    if not aggregated:
        return "No data to display."

    lines = []
    header = (
        f"{'Metric':<22} {'Perspective':<12} "
        f"{'SD%':>5} {'D%':>5} {'N%':>5} {'A%':>5} {'SA%':>5} "
        f"{'T2B%':>6} {'B2B%':>6} {'Mean':>5} {'N':>4} {'Conf':>5}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for am in aggregated:
        d = am.distribution
        metric_display = am.metric_name.replace("_", " ").title()
        line = (
            f"{metric_display:<22} {am.perspective:<12} "
            f"{d[0]:>5.1f} {d[1]:>5.1f} {d[2]:>5.1f} {d[3]:>5.1f} {d[4]:>5.1f} "
            f"{am.top2_box:>6.1f} {am.bottom2_box:>6.1f} {am.mean:>5.2f} {am.n:>4} {am.avg_confidence:>5.2f}"
        )
        lines.append(line)

    return "\n".join(lines)
