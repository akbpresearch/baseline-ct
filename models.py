from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ContentItem:
    url: str
    source_type: str  # "article", "search", "reddit", "youtube"
    title: str
    text: str
    scraped_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    word_count: int = 0

    def __post_init__(self):
        if self.word_count == 0:
            self.word_count = len(self.text.split())


@dataclass
class MetricScore:
    metric_name: str  # purchase_intent, uniqueness, believability, value_perception
    author_score: int  # 1-5 Likert
    author_confidence: float  # 0.0-1.0
    author_reasoning: str
    aggregate_score: int  # 1-5 Likert
    aggregate_confidence: float  # 0.0-1.0
    aggregate_reasoning: str


@dataclass
class ExtractionResult:
    content_item: ContentItem
    metrics: list  # list of MetricScore
    overall_sentiment: str
    is_relevant: bool
