#!/usr/bin/env python3
"""Streamlit app for GroundSource-style product concept testing."""

import json
import logging
import os
import re
import time

import pandas as pd
import streamlit as st

from scraper import scrape_all
from extractor import extract_all, save_scraped_cache, load_scraped_cache
from aggregator import aggregate_results, AggregatedMetric, METRIC_NAMES
from exporter import export_summary_csv, export_raw_csv, export_final_json
from models import ExtractionResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

METRIC_LABELS = {
    "purchase_intent": "Purchase Intent",
    "uniqueness": "Uniqueness",
    "believability": "Believability",
    "value_perception": "Value Perception",
}

SENTIMENT_COLORS = {
    "very_positive": "#22c55e",
    "positive": "#86efac",
    "neutral": "#fbbf24",
    "negative": "#fca5a5",
    "very_negative": "#ef4444",
}


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s-]+", "_", text)
    return text[:50]


def _build_summary_df(aggregated: list[AggregatedMetric]) -> pd.DataFrame:
    rows = []
    for am in aggregated:
        d = am.distribution
        rows.append({
            "Metric": METRIC_LABELS.get(am.metric_name, am.metric_name),
            "Perspective": am.perspective.title(),
            "SD %": d[0],
            "D %": d[1],
            "N %": d[2],
            "A %": d[3],
            "SA %": d[4],
            "Top 2 Box %": am.top2_box,
            "Bottom 2 Box %": am.bottom2_box,
            "Mean": am.mean,
            "N": am.n,
            "Avg Confidence": am.avg_confidence,
        })
    return pd.DataFrame(rows)


def _build_raw_df(results: list[ExtractionResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "URL": r.content_item.url,
            "Source": r.content_item.source_type,
            "Title": r.content_item.title[:60],
            "Words": r.content_item.word_count,
            "Relevant": r.is_relevant,
            "Sentiment": r.overall_sentiment,
        }
        for m in r.metrics:
            label = METRIC_LABELS.get(m.metric_name, m.metric_name)
            row[f"{label} (A)"] = m.author_score
            row[f"{label} Conf (A)"] = m.author_confidence
            row[f"{label} (Agg)"] = m.aggregate_score
            row[f"{label} Conf (Agg)"] = m.aggregate_confidence
        rows.append(row)
    return pd.DataFrame(rows)


def _build_final_json(aggregated: list[AggregatedMetric],
                      results: list[ExtractionResult], product_name: str) -> dict:
    total = len(results)
    relevant = sum(1 for r in results if r.is_relevant)
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
    raw_items = []
    for r in results:
        item_scores = {}
        for m in r.metrics:
            item_scores[m.metric_name] = {
                "author": {"score": m.author_score, "confidence": m.author_confidence, "reasoning": m.author_reasoning},
                "aggregate": {"score": m.aggregate_score, "confidence": m.aggregate_confidence, "reasoning": m.aggregate_reasoning},
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
    return {"product_name": product_name, "summary": {"total_items": total, "relevant_items": relevant, "metrics": metrics}, "items": raw_items}


def _render_metric_cards(aggregated: list[AggregatedMetric]):
    """Render metric score cards in a 2x2 grid."""
    # Group by metric
    by_metric = {}
    for am in aggregated:
        by_metric.setdefault(am.metric_name, {})[am.perspective] = am

    cols = st.columns(2)
    for i, metric_name in enumerate(METRIC_NAMES):
        if metric_name not in by_metric:
            continue
        data = by_metric[metric_name]
        author = data.get("author")
        agg = data.get("aggregate")
        with cols[i % 2]:
            label = METRIC_LABELS.get(metric_name, metric_name)
            st.markdown(f"#### {label}")

            mc1, mc2 = st.columns(2)
            with mc1:
                if author:
                    t2b_color = "#22c55e" if author.top2_box >= 60 else "#fbbf24" if author.top2_box >= 30 else "#ef4444"
                    st.metric("Author Mean", f"{author.mean:.2f}", f"T2B: {author.top2_box:.0f}%")
                    st.caption(f"Confidence: {author.avg_confidence:.2f} | N={author.n}")
            with mc2:
                if agg:
                    st.metric("Aggregate Mean", f"{agg.mean:.2f}", f"T2B: {agg.top2_box:.0f}%")
                    st.caption(f"Confidence: {agg.avg_confidence:.2f} | N={agg.n}")

            # Distribution bar
            if author:
                dist_data = pd.DataFrame({
                    "Score": ["1 (SD)", "2 (D)", "3 (N)", "4 (A)", "5 (SA)"],
                    "Author %": author.distribution,
                    "Aggregate %": agg.distribution if agg else [0]*5,
                })
                st.bar_chart(dist_data, x="Score", y=["Author %", "Aggregate %"], height=180)
            st.divider()


def run_pipeline_ui(product_name: str, category: str | None, description: str | None,
                    max_articles: int, max_results_per_query: int, cached_file) -> None:
    """Run pipeline with Streamlit progress UI."""
    slug = _slugify(product_name)
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    cache_path = os.path.join(output_dir, f"{slug}_scraped.json")
    summary_path = os.path.join(output_dir, f"{slug}_summary.csv")
    raw_path = os.path.join(output_dir, f"{slug}_raw.csv")
    json_path = os.path.join(output_dir, f"{slug}_final.json")

    progress = st.progress(0, text="Starting pipeline...")

    # ── Step 1: Scrape ──
    if cached_file is not None:
        progress.progress(10, text="Loading cached content...")
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(cached_file.getvalue())
            tmp_path = tmp.name
        content_items = load_scraped_cache(tmp_path)
        os.unlink(tmp_path)
        st.info(f"Loaded {len(content_items)} cached items")
    else:
        progress.progress(5, text=f'Searching for content about "{product_name}"...')
        with st.status("Scraping content...", expanded=True) as status:
            st.write("Running DuckDuckGo, Reddit & YouTube searches...")
            content_items = scrape_all(
                product_name=product_name,
                category=category,
                max_articles=max_articles,
                max_results_per_query=max_results_per_query,
                timeout=15,
                delay=1.5,
            )
            if not content_items:
                st.error("No content found. Try a different product name.")
                return
            save_scraped_cache(content_items, cache_path)
            status.update(label=f"Scraped {len(content_items)} items", state="complete")

    progress.progress(25, text="Extracting metrics with Claude...")

    # ── Step 2: Extract ──
    with st.status("Extracting metrics with Claude...", expanded=True) as status:
        start = time.time()
        results = extract_all(
            content_items=content_items,
            product_name=product_name,
            category=category or "",
            description=description or "",
            model="claude-sonnet-4-0",
            batch_size=5,
        )
        elapsed = time.time() - start
        relevant_count = sum(1 for r in results if r.is_relevant)
        status.update(label=f"Extracted {len(results)} items ({relevant_count} relevant) in {elapsed:.0f}s", state="complete")

    progress.progress(75, text="Aggregating results...")

    # ── Step 3: Aggregate ──
    aggregated = aggregate_results(results, confidence_threshold=st.session_state.get("confidence_threshold", 0.0))

    progress.progress(90, text="Exporting files...")

    # ── Step 4: Export ──
    export_summary_csv(aggregated, summary_path)
    export_raw_csv(results, raw_path)
    export_final_json(aggregated, results, product_name, json_path)
    final_json = _build_final_json(aggregated, results, product_name)

    progress.progress(100, text="Done!")

    # Store results in session state
    st.session_state["results"] = results
    st.session_state["aggregated"] = aggregated
    st.session_state["final_json"] = final_json
    st.session_state["product_name"] = product_name
    st.session_state["paths"] = {
        "summary": summary_path,
        "raw": raw_path,
        "json": json_path,
        "cache": cache_path,
    }


def render_results():
    """Render results from session state."""
    if "results" not in st.session_state:
        return

    results = st.session_state["results"]
    product_name = st.session_state["product_name"]
    paths = st.session_state["paths"]

    # Re-aggregate with current slider value so changes take effect live
    current_threshold = st.session_state.get("confidence_threshold", 0.0)
    aggregated = aggregate_results(results, confidence_threshold=current_threshold)
    final_json = _build_final_json(aggregated, results, product_name)

    total = len(results)
    relevant = sum(1 for r in results if r.is_relevant)

    st.markdown("---")
    st.header(f"Results: {product_name}")

    # Key stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Items Scraped", total)
    c2.metric("Relevant Items", relevant)
    c3.metric("Irrelevant / Filtered", total - relevant)

    # Source breakdown
    source_counts = {}
    for r in results:
        src = r.content_item.source_type
        source_counts[src] = source_counts.get(src, 0) + 1
    if source_counts:
        src_cols = st.columns(len(source_counts))
        for col, (src, count) in zip(src_cols, sorted(source_counts.items())):
            col.metric(f"{src.title()} Sources", count)

    # Metric cards
    st.subheader("Metric Scores")
    if aggregated:
        _render_metric_cards(aggregated)
    else:
        st.warning("No relevant content found to aggregate.")

    # Summary table
    st.subheader("Summary Table")
    if aggregated:
        summary_df = _build_summary_df(aggregated)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Relevant items JSON view
    relevant_results = [r for r in results if r.is_relevant]
    if relevant_results:
        st.subheader(f"Relevant Items ({len(relevant_results)})")
        for r in relevant_results:
            scores = {}
            for m in r.metrics:
                scores[m.metric_name] = {
                    "author": {"score": m.author_score, "confidence": m.author_confidence, "reasoning": m.author_reasoning},
                    "aggregate": {"score": m.aggregate_score, "confidence": m.aggregate_confidence, "reasoning": m.aggregate_reasoning},
                }
            with st.expander(f"[{r.content_item.source_type}] {r.content_item.title[:80]}"):
                st.json({
                    "url": r.content_item.url,
                    "source_type": r.content_item.source_type,
                    "word_count": r.content_item.word_count,
                    "overall_sentiment": r.overall_sentiment,
                    "scores": scores,
                })

    # Raw data
    st.subheader("Per-Item Scores")
    raw_df = _build_raw_df(results)
    st.dataframe(raw_df, use_container_width=True, hide_index=True)

    # Downloads
    st.subheader("Downloads")
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        if os.path.exists(paths["summary"]):
            with open(paths["summary"]) as f:
                st.download_button("Summary CSV", f.read(), file_name=os.path.basename(paths["summary"]), mime="text/csv")
    with dc2:
        if os.path.exists(paths["raw"]):
            with open(paths["raw"]) as f:
                st.download_button("Raw CSV", f.read(), file_name=os.path.basename(paths["raw"]), mime="text/csv")
    with dc3:
        json_str = json.dumps(final_json, indent=2)
        st.download_button("Final JSON", json_str, file_name=os.path.basename(paths["json"]), mime="application/json")

    # Expandable JSON preview
    with st.expander("Preview Final JSON"):
        st.json(final_json)


def main():
    st.set_page_config(page_title="Product Concept Tester", page_icon="🔬", layout="wide")

    st.title("Product Concept Testing Pipeline")
    st.caption("GroundSource-style LLM extraction from online content")

    # Sidebar config
    with st.sidebar:
        st.header("Configuration")

        api_key = st.text_input("Anthropic API Key", type="password",
                                value=os.environ.get("ANTHROPIC_API_KEY", ""),
                                help="Set ANTHROPIC_API_KEY env var or paste here")

        st.divider()

        input_mode = st.radio("Input mode", ["Named Product", "Product Concept"],
                              help="Test a specific product or describe a concept")

        max_articles = st.slider("Max articles to scrape", 5, 100, 20)
        max_results_per_query = st.slider("Results per search query", 5, 50, 20,
                                           help="More results per query = more unique URLs before dedup")
        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.3, step=0.05,
                                          help="Scores below this confidence are excluded from aggregation",
                                          key="confidence_threshold")

        st.divider()
        st.markdown("**Re-use cached scrape**")
        cached_file = st.file_uploader("Upload _scraped.json", type="json",
                                        help="Skip scraping, re-run extraction on cached data")

    # Main input area
    if input_mode == "Named Product":
        col1, col2 = st.columns([2, 1])
        with col1:
            product_name = st.text_input("Product name",
                                          placeholder="e.g. Stanley Quencher H2.0 Tumbler")
        with col2:
            category = st.text_input("Category (optional)",
                                      placeholder="e.g. drinkware, tech, beauty")
        description = st.text_input("Brief description (optional)",
                                     placeholder="e.g. 40oz insulated tumbler that keeps drinks cold")
    else:
        st.markdown("**Describe your product concept** and we'll search for related market content.")
        product_name = st.text_input("Product / concept name",
                                      placeholder="e.g. Greek Yogurt Parfait Protein Bites",
                                      help="A descriptive name helps find relevant market content")
        category = st.text_input("Category",
                                  placeholder="e.g. healthy snacks")

        description = st.text_area("Product description & characteristics",
                                    placeholder="Describe the product concept:\n"
                                                "- What problem does it solve?\n"
                                                "- Key claims (protein, sugar, ingredients)\n"
                                                "- Form factor and use case",
                                    height=150)

    # Run button
    if st.button("Run Concept Test", type="primary", use_container_width=True):
        if not product_name:
            st.error("Product name is required.")
            return
        if not api_key:
            st.error("Anthropic API key is required. Set it in the sidebar or as ANTHROPIC_API_KEY env var.")
            return

        os.environ["ANTHROPIC_API_KEY"] = api_key

        run_pipeline_ui(
            product_name=product_name,
            category=category or None,
            description=description or None,
            max_articles=max_articles,
            max_results_per_query=max_results_per_query,
            cached_file=cached_file,
        )

    # Always render results if they exist
    render_results()


if __name__ == "__main__":
    main()
