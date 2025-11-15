"""
news_loader.py
---------------
This module connects to the Perplexity Sonar API to fetch recent company news headlines
and store them locally as JSONL files. It serves as the "unstructured" ingestion step
for your Agentic AI Equity Analyst MVP.

MVP GOAL:
- Input: Company ticker symbol (e.g., AAPL)
- Output: JSON lines file with news headlines, URLs, snippets, and dates
- Usage: Can later feed into RAG or reporting agents for contextual analysis
"""

from __future__ import annotations
import os
import json
import time
import pathlib
from dataclasses import dataclass
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import Perplexity API client (official SDK)
# Docs: https://docs.perplexity.ai/guides/search-quickstart
from perplexity import Perplexity


# Define a data structure for one news record
@dataclass
class NewsItem:
    """Represents one news article entry returned by Sonar."""
    title: str
    url: str
    snippet: str | None
    date: str | None       # publication date (if available)
    source: str | None     # domain name (e.g., 'reuters.com')
    ticker: str            # stock ticker for traceability


class SonarNewsClient:
    """
    Wraps Perplexity's Sonar Search API for finance-related queries.
    Handles building the search prompt, sending the request,
    and converting results into a structured Python list.
    """
    def __init__(self, model: str = "sonar-pro", max_results: int = 8):
        """
        Args:
            model: Which Sonar model to use ("sonar" or "sonar-pro")
            max_results: Maximum number of news results to return
        """
        # Load API keys from .env file
        load_dotenv()
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise RuntimeError("Missing PPLX_API_KEY in environment. Add it to your .env file.")

        # Initialize Perplexity client (auto-uses env key)
        self.client = Perplexity()
        self.model = model
        self.max_results = max_results

    def build_query(self, ticker: str, time_range: str = "1m") -> str:
        """
        Build a natural-language search query for Sonar.
        Sonar understands context prompts, so this guides it to finance-specific results.
        """
        return (
            f"Recent market-moving news for {ticker}. "
            f"Focus on earnings, guidance, M&A, regulation, and management commentary "
            f"in the last {time_range}. Return diverse, reputable sources."
        )

    def search_news(self, ticker: str, time_range: str = "1m") -> List[NewsItem]:
        """
        Send the search query to the Sonar API and parse the results.

        Args:
            ticker: e.g., 'AAPL'
            time_range: rough recency hint ('24h', '1w', '1m', etc.)

        Returns:
            List[NewsItem] - each item includes title, URL, snippet, and metadata.
        """
        query = self.build_query(ticker, time_range)

        # Use the Perplexity Search API to fetch results
        # The response includes a list of ranked results with metadata
        search = self.client.search.create(
            query=query,
            max_results=self.max_results,
            max_tokens_per_page=1024,  # allows longer snippets per result
        )

        # Convert raw results into structured NewsItem dataclasses
        items: List[NewsItem] = []
        for r in getattr(search, "results", []) or []:
            items.append(
                NewsItem(
                    title=getattr(r, "title", None) or "",
                    url=getattr(r, "url", None) or "",
                    snippet=getattr(r, "snippet", None),
                    date=getattr(r, "date", None),
                    source=(getattr(r, "url", "") or "").split("/")[2]
                    if getattr(r, "url", None)
                    else None,
                    ticker=ticker.upper(),
                )
            )

        return items


def save_jsonl(records: List[Dict[str, Any]], path: str):
    """
    Utility function to save a list of dicts as a JSONL (JSON Lines) file.
    Each line = one JSON object.
    """
    pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def fetch_and_save_news(
    ticker: str,
    out_dir: str = "data/raw/news",
    time_range: str = "1m",
    model: str = "sonar-pro",
    max_results: int = 8
) -> str:
    """
    Convenience function that combines all steps:
    - Create client
    - Fetch news for given ticker
    - Save results to a timestamped JSONL file

    Returns:
        Path to the output file written.
    """
    client = SonarNewsClient(model=model, max_results=max_results)
    items = client.search_news(ticker, time_range=time_range)

    # Create output path with timestamp
    ts = time.strftime("%Y-%m-%d")
    out_path = os.path.join(out_dir, f"{ticker.upper()}_{ts}.jsonl")

    # Save results
    save_jsonl([item.__dict__ for item in items], out_path)
    return out_path


# Command-line entrypoint (lets you run this script standalone)
if __name__ == "__main__":
    import argparse

    # Define CLI arguments for flexibility
    parser = argparse.ArgumentParser(description="Fetch company news via Sonar API")
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--time-range", default="1m", choices=["24h", "1w", "1m", "3m", "1y"])
    parser.add_argument("--model", default="sonar-pro", help="Sonar model to use")
    parser.add_argument("--max-results", type=int, default=8, help="Max number of news items")
    parser.add_argument("--out-dir", default="data/raw/news", help="Output directory")

    args = parser.parse_args()

    # Run the fetch + save process
    path = fetch_and_save_news(
        ticker=args.ticker,
        out_dir=args.out_dir,
        time_range=args.time_range,
        model=args.model,
        max_results=args.max_results,
    )

    print(f"✅ News data saved to: {path}")
