"""
Daily News Briefing Pipeline
Fetches configured RSS/Reddit sources -> dedupes -> Gemini filters+summarizes -> Telegram push
Run daily via GitHub Actions cron.
"""

import json
import os
import hashlib
import requests
import feedparser
from datetime import datetime, timezone
from xml.sax.saxutils import escape

CONFIG_PATH = "sources_config.json"
SEEN_PATH = "seen_ids.json"
ARCHIVE_PATH = "briefing_archive.json"
FEED_OUTPUT_PATH = "briefing_feed.xml"
MAX_ARCHIVE_ENTRIES = 60  # ~2 months of daily entries kept in the feed

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
# Set by the GitHub Actions workflow to build the raw feed URL for the Atom <link>
FEED_PUBLIC_URL = os.environ.get("FEED_PUBLIC_URL", "")

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-flash-latest:generateContent?key={GEMINI_API_KEY}"
)

# feedparser uses urllib by default with a generic UA - some news sites block that.
# A browser-like UA fixes most of these silently.
FEEDPARSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def entry_id(entry):
    key = entry.get("link") or entry.get("title", "")
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def fetch_category_entries(category_key, category_cfg, seen_ids, max_per_source=8, max_per_category=20):
    """Fetch entries for one category, skipping already-seen items. Caps the total per
    category so the Gemini prompt doesn't balloon in size (which risks timeouts)."""
    collected = []
    for source in category_cfg["sources"]:
        url = source["url"]
        if not url or "TBD" in url:
            continue
        try:
            feed = feedparser.parse(url, agent=FEEDPARSER_USER_AGENT)
        except Exception as e:
            print(f"[WARN] Failed to fetch {source['name']}: {e}")
            continue

        for entry in feed.entries[:max_per_source]:
            eid = entry_id(entry)
            if eid in seen_ids:
                continue
            collected.append({
                "id": eid,
                "title": entry.get("title", "").strip(),
                "snippet": (entry.get("summary", "") or "")[:300].strip(),
                "link": entry.get("link", ""),
                "source": source["name"],
            })
    return collected[:max_per_category]


def build_gemini_prompt(all_category_items, category_meta):
    """Build one big prompt covering all categories, asking for filtered+summarized JSON back."""
    sections = []
    for cat_key, items in all_category_items.items():
        if not items:
            continue
        meta = category_meta[cat_key]
        lines = [f"Category: {cat_key}", f"Guidance: {meta['filter_hint']}", "Items:"]
        for i, item in enumerate(items):
            lines.append(f"  [{i}] {item['title']} | snippet: {item['snippet']} | source: {item['source']}")
        sections.append("\n".join(lines))

    joined = "\n\n".join(sections)

    prompt = f"""You are curating a personal daily news briefing for a busy individual in Bangalore, India,
who is a Python backend developer transitioning into cybersecurity (SOC Analyst path) and also
building an AI-generated YouTube content business. He explicitly does NOT want celebrity gossip,
entertainment/brain-rot content, or generic filler news.

For each category below, select the {{max_items}} MOST relevant and important items based on the
category's guidance. Write a single concise one-sentence summary per item (plain, no fluff, no hype).
Skip items that are trivial, redundant, or off-topic per the guidance - if there are fewer than
{{max_items}} genuinely relevant items in a category, return fewer rather than padding with filler.

IMPORTANT: Do NOT retype the title or invent a link. Just return the index number of the item you
selected (the number in square brackets before each item below) plus your one-sentence summary.

Return ONLY valid JSON in this exact structure, no markdown fences, no preamble:

{{{{
  "category_key": [
    {{{{"index": 0, "summary": "..."}}}}
  ]
}}}}

Here is the data:

{joined}
"""
    return prompt


def call_gemini(prompt, max_items, max_retries=2):
    prompt = prompt.replace("{max_items}", str(max_items))
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3},
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(GEMINI_URL, json=body, timeout=150)
            if not resp.ok:
                print(f"[ERROR] Gemini API returned {resp.status_code}: {resp.text[:500]}")
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(text)
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_error = e
            print(f"[WARN] Gemini call attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                print("[INFO] Retrying...")
    raise last_error


def resolve_selection(selection, all_category_items):
    """Map Gemini's {index, summary} picks back to the real fetched item (title/source/link),
    so links are always the genuine fetched URL - never something the model retyped."""
    resolved = {}
    for cat_key, picks in selection.items():
        items = all_category_items.get(cat_key, [])
        resolved_items = []
        for pick in picks:
            idx = pick.get("index")
            if idx is None or idx < 0 or idx >= len(items):
                continue  # skip anything out of range rather than guess
            original = items[idx]
            resolved_items.append({
                "title": original["title"],
                "summary": pick.get("summary", "").strip(),
                "source": original["source"],
                "link": original["link"],
            })
        resolved[cat_key] = resolved_items
    return resolved


def format_briefing(result, category_meta):
    date_str = datetime.now(timezone.utc).strftime("%d %b %Y")
    lines = [f"📰 Daily Briefing — {date_str}\n"]
    for cat_key, meta in category_meta.items():
        items = result.get(cat_key, [])
        if not items:
            continue
        lines.append(meta["label"])
        for item in items:
            lines.append(f"• {item['title']} — {item['summary']} ({item['source']})")
            if item.get("link"):
                lines.append(f"  🔗 {item['link']}")
        lines.append("")
    return "\n".join(lines).strip()


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    # Telegram hard limit is 4096 chars - split if needed
    chunks = [message[i:i + 4000] for i in range(0, len(message), 4000)] or [message]
    for chunk in chunks:
        resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk})
        resp.raise_for_status()


def update_archive_and_build_feed(message_text, date_str):
    """Append today's briefing to a rolling archive, then regenerate the Atom feed file
    so any RSS reader (e.g. Feedly) can subscribe to your own filtered/summarized output."""
    archive = load_json(ARCHIVE_PATH, [])

    entry_id = f"briefing-{date_str.replace(' ', '-')}"
    new_entry = {
        "id": entry_id,
        "title": f"Daily Briefing — {date_str}",
        "content": message_text,
        "published": datetime.now(timezone.utc).isoformat(),
    }

    # avoid duplicate entry if script somehow runs twice same day
    archive = [e for e in archive if e["id"] != entry_id]
    archive.append(new_entry)
    archive = archive[-MAX_ARCHIVE_ENTRIES:]

    save_json(ARCHIVE_PATH, archive)
    write_atom_feed(archive)


def write_atom_feed(archive):
    """Hand-build a minimal, valid Atom feed from the archive list - no external library needed."""
    self_link = FEED_PUBLIC_URL or "https://example.com/briefing_feed.xml"
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    entries_xml = []
    # newest first, standard for feeds
    for entry in reversed(archive):
        content_escaped = escape(entry["content"])
        entries_xml.append(f"""  <entry>
    <id>urn:briefing:{escape(entry['id'])}</id>
    <title>{escape(entry['title'])}</title>
    <updated>{entry['published']}</updated>
    <published>{entry['published']}</published>
    <content type="text">{content_escaped}</content>
  </entry>""")

    feed_xml = f"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <id>urn:briefing:personal-daily-briefing</id>
  <title>My Daily Briefing</title>
  <link rel="self" href="{escape(self_link)}"/>
  <updated>{updated}</updated>
{chr(10).join(entries_xml)}
</feed>
"""
    with open(FEED_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(feed_xml)


def send_telegram_error(stage, error):
    """Best-effort notification so a failure is never silent - keeps it short, no stack trace."""
    short_error = str(error)[:200]
    message = (
        f"⚠️ Daily Briefing FAILED\n\n"
        f"Stage: {stage}\n"
        f"Error: {short_error}\n\n"
        f"Check GitHub Actions logs for full details."
    )
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=15)
    except Exception as notify_error:
        # If even the error notification fails, just log it - nothing more we can do.
        print(f"[ERROR] Could not send failure notification to Telegram: {notify_error}")


def main():
    stage = "startup"
    try:
        stage = "loading config"
        config = load_json(CONFIG_PATH, {})
        seen_ids = set(load_json(SEEN_PATH, []))

        categories = config["categories"]
        max_items = config["briefing_config"].get("items_per_category", 4)

        stage = "fetching sources"
        all_category_items = {}
        for cat_key, cat_cfg in categories.items():
            items = fetch_category_entries(cat_key, cat_cfg, seen_ids)
            all_category_items[cat_key] = items
            print(f"[INFO] {cat_key}: {len(items)} new items fetched")

        if not any(all_category_items.values()):
            print("[INFO] No new items across all categories, skipping send.")
            return

        stage = "building Gemini prompt"
        prompt = build_gemini_prompt(all_category_items, categories)

        stage = "calling Gemini API"
        selection = call_gemini(prompt, max_items)

        stage = "resolving selected items"
        result = resolve_selection(selection, all_category_items)

        stage = "formatting briefing"
        message = format_briefing(result, categories)

        stage = "sending to Telegram"
        send_telegram(message)

        stage = "updating archive and RSS feed"
        date_str = datetime.now(timezone.utc).strftime("%d %b %Y")
        update_archive_and_build_feed(message, date_str)

        stage = "saving seen_ids"
        for items in all_category_items.values():
            for item in items:
                seen_ids.add(item["id"])
        save_json(SEEN_PATH, list(seen_ids)[-2000:])

        print("[INFO] Briefing sent successfully.")

    except Exception as e:
        print(f"[ERROR] Failed at stage '{stage}': {e}")
        send_telegram_error(stage, e)
        raise  # re-raise so the GitHub Actions run still shows as failed (red X)


if __name__ == "__main__":
    main()