import json
import logging
import os
from dataclasses import asdict

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from models import ContentItem, MetricScore, ExtractionResult

logger = logging.getLogger(__name__)

EXTRACTION_TOOL = {
    "name": "record_concept_test_scores",
    "description": (
        "Record structured concept test scores extracted from consumer content. "
        "Call this tool exactly once per content item with all metrics filled in."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "is_relevant": {
                "type": "boolean",
                "description": "Whether the content is relevant to the product being tested"
            },
            "overall_sentiment": {
                "type": "string",
                "enum": ["very_negative", "negative", "neutral", "positive", "very_positive"],
                "description": "Overall sentiment of the content toward the product"
            },
            "purchase_intent": {
                "type": "object",
                "description": "Would the author/audience buy this product? (1=Definitely Not, 5=Definitely Yes)",
                "properties": {
                    "author_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "author_confidence": {"type": "number", "description": "0.0-1.0"},
                    "author_reasoning": {"type": "string"},
                    "aggregate_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "aggregate_confidence": {"type": "number", "description": "0.0-1.0"},
                    "aggregate_reasoning": {"type": "string"}
                },
                "required": ["author_score", "author_confidence", "author_reasoning",
                             "aggregate_score", "aggregate_confidence", "aggregate_reasoning"]
            },
            "uniqueness": {
                "type": "object",
                "description": "How unique/differentiated is this product? (1=Not At All, 5=Extremely Unique)",
                "properties": {
                    "author_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "author_confidence": {"type": "number", "description": "0.0-1.0"},
                    "author_reasoning": {"type": "string"},
                    "aggregate_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "aggregate_confidence": {"type": "number", "description": "0.0-1.0"},
                    "aggregate_reasoning": {"type": "string"}
                },
                "required": ["author_score", "author_confidence", "author_reasoning",
                             "aggregate_score", "aggregate_confidence", "aggregate_reasoning"]
            },
            "believability": {
                "type": "object",
                "description": "Are the product claims believable? (1=Not At All, 5=Completely Believable)",
                "properties": {
                    "author_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "author_confidence": {"type": "number", "description": "0.0-1.0"},
                    "author_reasoning": {"type": "string"},
                    "aggregate_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "aggregate_confidence": {"type": "number", "description": "0.0-1.0"},
                    "aggregate_reasoning": {"type": "string"}
                },
                "required": ["author_score", "author_confidence", "author_reasoning",
                             "aggregate_score", "aggregate_confidence", "aggregate_reasoning"]
            },
            "value_perception": {
                "type": "object",
                "description": "Is this product good value for money? (1=Very Poor Value, 5=Excellent Value)",
                "properties": {
                    "author_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "author_confidence": {"type": "number", "description": "0.0-1.0"},
                    "author_reasoning": {"type": "string"},
                    "aggregate_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "aggregate_confidence": {"type": "number", "description": "0.0-1.0"},
                    "aggregate_reasoning": {"type": "string"}
                },
                "required": ["author_score", "author_confidence", "author_reasoning",
                             "aggregate_score", "aggregate_confidence", "aggregate_reasoning"]
            }
        },
        "required": ["is_relevant", "overall_sentiment",
                     "purchase_intent", "uniqueness", "believability", "value_perception"]
    }
}

SYSTEM_PROMPT_TEMPLATE = """You are an expert market research analyst conducting a product concept test.

PRODUCT UNDER TEST:
- Name: {product_name}
- Category: {category}
- Description: {description}

YOUR TASK:
Analyze the provided consumer content (review, discussion, article) and extract structured concept test metrics on a 1-5 Likert scale.

METRICS TO SCORE:
1. Purchase Intent: Would the author/audience buy this product? (1=Definitely Not, 5=Definitely Yes)
2. Uniqueness: How unique/differentiated is this product vs alternatives? (1=Not At All Unique, 5=Extremely Unique)
3. Believability: Are the product's claims and promises believable? (1=Not At All Believable, 5=Completely Believable)
4. Value Perception: Is this product good value for the money? (1=Very Poor Value, 5=Excellent Value)

TWO PERSPECTIVES FOR EACH METRIC:
- Author perspective: What does the specific author of this content think? Score based on their explicit statements.
- Aggregate perspective: Based on this content, what would the broader target audience likely think? Consider the general consumer sentiment this content represents.

CONFIDENCE CALIBRATION:
- 0.0: No signal at all for this metric in the content
- 0.1-0.3: Weak/indirect signal, mostly inferring
- 0.4-0.6: Moderate signal, some direct evidence
- 0.7-0.9: Strong signal, clear direct evidence
- 1.0: Extremely explicit and unambiguous signal

CRITICAL RULES:
- If the content is NOT relevant to the product (e.g., about a completely different product), set is_relevant=false and score all metrics as 3 with confidence 0.0
- Always cite specific phrases or evidence from the content in your reasoning
- Be calibrated: not everything is a 5 or a 1. Use the full range.
- The author perspective should reflect what THIS specific person thinks
- The aggregate perspective should reflect what the BROADER market likely thinks based on signals in this content"""


def _build_system_prompt(product_name: str, category: str, description: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        product_name=product_name,
        category=category or "general",
        description=description or "No description provided",
    )


def _parse_tool_result(tool_input: dict, content_item: ContentItem) -> ExtractionResult:
    """Parse the tool_use response into an ExtractionResult."""
    metric_names = ["purchase_intent", "uniqueness", "believability", "value_perception"]
    metrics = []
    for name in metric_names:
        m = tool_input[name]
        metrics.append(MetricScore(
            metric_name=name,
            author_score=m["author_score"],
            author_confidence=m["author_confidence"],
            author_reasoning=m["author_reasoning"],
            aggregate_score=m["aggregate_score"],
            aggregate_confidence=m["aggregate_confidence"],
            aggregate_reasoning=m["aggregate_reasoning"],
        ))
    return ExtractionResult(
        content_item=content_item,
        metrics=metrics,
        overall_sentiment=tool_input["overall_sentiment"],
        is_relevant=tool_input["is_relevant"],
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError)),
    before_sleep=lambda retry_state: logger.warning(
        f"Rate limited, retrying in {retry_state.next_action.sleep}s..."
    ),
)
def _extract_single(client: anthropic.Anthropic, system_prompt: str,
                     content_item: ContentItem, model: str) -> ExtractionResult:
    """Extract metrics from a single content item."""
    user_message = (
        f"Content source: {content_item.source_type}\n"
        f"URL: {content_item.url}\n"
        f"Title: {content_item.title}\n\n"
        f"--- CONTENT ---\n{content_item.text}"
    )

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=0.0,
        system=system_prompt,
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "record_concept_test_scores"},
        messages=[{"role": "user", "content": user_message}],
    )

    for block in response.content:
        if block.type == "tool_use":
            return _parse_tool_result(block.input, content_item)

    raise ValueError("No tool_use block in response")


def extract_all(content_items: list[ContentItem], product_name: str,
                category: str, description: str, model: str = "claude-sonnet-4-0",
                batch_size: int = 5) -> list[ExtractionResult]:
    """Extract metrics from all content items."""
    client = anthropic.Anthropic()
    system_prompt = _build_system_prompt(product_name, category, description)
    results: list[ExtractionResult] = []

    total = len(content_items)
    for i, item in enumerate(content_items):
        try:
            logger.info(f"Extracting [{i+1}/{total}]: {item.url[:80]}")
            result = _extract_single(client, system_prompt, item, model)
            results.append(result)
        except Exception as e:
            logger.error(f"Extraction failed for {item.url}: {e}")
            # Create a default irrelevant result on failure
            metrics = [
                MetricScore(
                    metric_name=name,
                    author_score=3, author_confidence=0.0,
                    author_reasoning="Extraction failed",
                    aggregate_score=3, aggregate_confidence=0.0,
                    aggregate_reasoning="Extraction failed",
                )
                for name in ["purchase_intent", "uniqueness", "believability", "value_perception"]
            ]
            results.append(ExtractionResult(
                content_item=item, metrics=metrics,
                overall_sentiment="neutral", is_relevant=False,
            ))

    return results


def save_scraped_cache(content_items: list[ContentItem], path: str):
    """Save scraped content as JSON for re-use."""
    data = [asdict(item) for item in content_items]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Cached {len(data)} items to {path}")


def load_scraped_cache(path: str) -> list[ContentItem]:
    """Load cached scraped content from JSON."""
    with open(path) as f:
        data = json.load(f)
    items = [ContentItem(**d) for d in data]
    logger.info(f"Loaded {len(items)} items from {path}")
    return items
