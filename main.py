#!/usr/bin/env python3
"""
Hacker News Digest Agent

Fetches articles from Hacker News front page, reads their content,
and creates a summary digest using the Strands Agents framework.
"""

from datetime import datetime
import httpx
from bs4 import BeautifulSoup
from strands import Agent, tool
from strands.models import BedrockModel


@tool
def fetch_webpage(url: str) -> str:
    """Fetch the content of a webpage and return it as text.

    Args:
        url: The URL of the webpage to fetch
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()

            # Get text content
            text = soup.get_text(separator="\n", strip=True)

            # Truncate if too long
            if len(text) > 15000:
                text = text[:15000] + "\n\n[Content truncated...]"

            return text
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"


def fetch_hn_front_page() -> list[dict]:
    """Fetch the front page of Hacker News and return article information."""
    url = "https://news.ycombinator.com/"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    with httpx.Client(timeout=15.0) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    articles = []

    # Find all article rows
    title_rows = soup.select("tr.athing")

    for row in title_rows:
        try:
            # Get the title and link - find the titleline span directly
            title_link = row.select_one("span.titleline > a")
            if not title_link:
                continue

            title = title_link.get_text(strip=True)
            link = title_link.get("href", "")

            # Skip internal HN links like "item?id=..."
            if link.startswith("item?"):
                link = f"https://news.ycombinator.com/{link}"

            # Get the subtext row for points and comments
            subtext_row = row.find_next_sibling("tr")
            points = 0
            comments = 0

            if subtext_row:
                score_span = subtext_row.select_one("span.score")
                if score_span:
                    score_text = score_span.get_text(strip=True)
                    points = int(score_text.split()[0])

                # Find comments link
                subline = subtext_row.select_one("td.subtext")
                if subline:
                    links = subline.select("a")
                    for a in links:
                        text = a.get_text(strip=True)
                        if "comment" in text:
                            try:
                                comments = int(text.split()[0])
                            except ValueError:
                                comments = 0

            articles.append({
                "title": title,
                "url": link,
                "points": points,
                "comments": comments
            })

        except Exception:
            continue

    return articles


def create_agent() -> Agent:
    """Create and configure the Strands agent using Amazon Bedrock."""
    model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
        max_tokens=4096,
        temperature=0.3,
    )

    agent = Agent(
        model=model,
        tools=[fetch_webpage],
        system_prompt="""You are a helpful assistant that summarizes web articles concisely.
When asked to summarize an article, provide:
1. A 2-3 sentence summary of the main points
2. Key takeaways or interesting facts
Keep summaries brief and informative."""
    )

    return agent


def main():
    """Main function to run the HN digest agent."""
    print("=" * 60)
    print("HACKER NEWS DAILY DIGEST")
    print("=" * 60)
    print()

    # Fetch HN front page articles
    print("Fetching Hacker News front page...")
    articles = fetch_hn_front_page()
    print(f"Found {len(articles)} articles\n")

    if not articles:
        print("No articles found. Exiting.")
        return

    # Create the agent
    agent = create_agent()

    # Process each article
    summaries = []

    num_articles = min(len(articles), 30)
    for i, article in enumerate(articles[:30], 1):  # Process top 30
        print(f"\n[{i}/{num_articles}] Processing: {article['title'][:60]}...")

        try:
            prompt = f"""Please fetch and summarize this article:
Title: {article['title']}
URL: {article['url']}

Use the fetch_webpage tool to get the article content, then provide a brief summary."""

            response = agent(prompt)

            summary = {
                "title": article["title"],
                "url": article["url"],
                "points": article["points"],
                "comments": article["comments"],
                "summary": str(response)
            }
            summaries.append(summary)
            print(f"   ✓ Summarized successfully")

        except Exception as e:
            print(f"   ✗ Skipped: {str(e)[:50]}")
            continue

    # Generate the final digest
    print("\n" + "=" * 60)
    print("GENERATING FINAL DIGEST")
    print("=" * 60 + "\n")

    digest_prompt = f"""Based on the following article summaries from Hacker News, create a cohesive daily digest.
Group related articles together if possible, and highlight the most significant stories.

Articles:
"""

    for s in summaries:
        digest_prompt += f"""
---
Title: {s['title']}
Points: {s['points']} | Comments: {s['comments']}
URL: {s['url']}
Summary: {s['summary']}
"""

    digest_prompt += """

Please create a well-formatted digest with:
1. Top Stories (2-3 most significant)
2. Tech/Development news
3. Business/Startup news
4. Other interesting reads
5. A brief overall summary of today's HN front page themes

Format it nicely for reading."""

    # Create a fresh agent for the digest to avoid context overflow
    digest_agent = create_agent()
    final_digest = str(digest_agent(digest_prompt))

    # Save digest to markdown file
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"hn_digest_{date_str}.md"

    with open(filename, "w") as f:
        f.write(f"# Hacker News Digest - {date_str}\n\n")
        f.write(final_digest)

    print(f"\n✓ Digest saved to: {filename}")

    print("\n" + "=" * 60)
    print("TODAY'S HACKER NEWS DIGEST")
    print("=" * 60)
    print(final_digest)
    print("\n" + "=" * 60)
    print("END OF DIGEST")
    print("=" * 60)


if __name__ == "__main__":
    main()
