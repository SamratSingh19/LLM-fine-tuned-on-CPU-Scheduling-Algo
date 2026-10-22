import os
import re
import wikipedia
import requests
from bs4 import BeautifulSoup

wikipedia.set_user_agent("SchedTutorProject/1.0 (student project; contact: example@example.com)")

RAW_DOCS_DIR = "data/raw_docs"


def safe_filename(title):
    # to turns a page title into a filesystem safe filename
    return re.sub(r'[^a-zA-Z0-9_-]', '_', title.strip())[:80]


def fetch_wikipedia(title, out_dir=RAW_DOCS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    try:
        page = wikipedia.page(title, auto_suggest=False)
    except wikipedia.DisambiguationError as e:
        print(f"  '{title}' is ambiguous, try one of: {e.options[:5]}")
        return
    except wikipedia.PageError:
        print(f"  No Wikipedia page found for '{title}'")
        return

    out_path = os.path.join(out_dir, f"wiki_{safe_filename(title)}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Source: Wikipedia — {page.title}\n{page.url}\n\n")
        f.write(page.content)
    print(f"  Saved: {out_path}")

def fetch_webpage(url, out_dir=RAW_DOCS_DIR):

    os.makedirs(out_dir, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (educational project)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
 
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
 
    text = soup.get_text(separator="\n")
    # Collapse multiple blank lines left behind after stripping tags.
    text = re.sub(r'\n\s*\n+', '\n\n', text).strip()
 
    title = soup.title.string if soup.title else url
    out_path = os.path.join(out_dir, f"web_{safe_filename(title)}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Source: {url}\n\n")
        f.write(text)
    print(f"  Saved: {out_path}")

if __name__ == "__main__":
    wiki_topics = [
        "Scheduling (computing)",
        "Round-robin scheduling",
        "Shortest job next",
        "Fixed-priority pre-emptive scheduling",
        "Multilevel feedback queue",
    ]

    print("Fetching Wikipedia articles..")
    for topic in wiki_topics:
        fetch_wikipedia(topic)
