import contextlib
import re
import shutil
import unittest
import uuid
from pathlib import Path

from app.x_list_summarizer import XListFetcher


@contextlib.contextmanager
def workspace_temp_directory():
    """Create disposable report storage under the workspace on Windows."""
    path = Path.cwd() / "tests" / f".tmp-report-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def tweet(tweet_id, text, author, *, links=(), card=None, media=None):
    """Return the complete legacy tweet shape used by report rendering."""
    return {
        "id": tweet_id,
        "text": text,
        "author": author,
        "links": list(links),
        "card": card,
        "media": list(media or ()),
        "likes": 1,
        "retweets": 2,
        "replies": 3,
        "quotes": 4,
        "bookmarks": 5,
    }


class LegacyReportSecurityTests(unittest.TestCase):
    def test_report_escapes_external_text_and_rejects_executable_urls(self):
        fetcher = XListFetcher.__new__(XListFetcher)
        fetcher.list_info = {
            "name": '<script>alert("list")</script>',
            "list_names": ['AI "Daily" <img src=x onerror=alert(1)>'],
            "owner": 'owner" onmouseover="alert(1)',
            "owner_name": '<b>Owner</b>',
            "member_count": '<img src=x onerror=alert(1)>',
            "profile_image_url": "javascript:alert(1)",
        }

        safe_link = "https://example.com/article?a=1&b=2"
        linked = tweet(
            "123",
            '<script>alert("tweet")</script>\n正常中文 and English',
            'alice" onmouseover="alert(1)',
            links=(safe_link,),
            card={
                "title": '<img src=x onerror=alert(1)> Article',
                "description": '<script>alert("description")</script>',
                "image": 'https://example.com/image.jpg" onerror="alert(1)',
            },
            media=[{"type": "photo", "url": "javascript:alert(1)"}],
        )
        unsafe_linked = tweet(
            "456",
            '<img src=x onerror=alert(1)>',
            "mallory",
            links=("javascript:alert(1)",),
        )
        standalone = tweet(
            "789",
            '<script>alert("standalone")</script>\n第二行',
            'bob" data-x="bad',
        )
        aggregated = {
            "by_link": [(safe_link, [linked]), ("javascript:alert(1)", [unsafe_linked])],
            "no_links": [standalone],
        }
        ai_summary = (
            f'{safe_link} :: <script>alert("llm")</script>\n'
            'javascript:alert(1) :: <img src=x onerror=alert(1)>'
        )

        with workspace_temp_directory() as directory:
            report = directory / "report.html"
            fetcher.generate_html_report(
                aggregated,
                ai_summary,
                report,
                tweet_count='<img src=x onerror=alert(1)>',
                ai_model='<b onclick="alert(1)">Model</b>',
            )
            rendered = report.read_text(encoding="utf-8")

        self.assertNotIn('<script>alert("tweet")</script>', rendered)
        self.assertNotIn('<script>alert("llm")</script>', rendered)
        self.assertNotIn('<img src=x onerror=alert(1)>', rendered)
        self.assertNotIn('href="javascript:', rendered.lower())
        self.assertNotIn('src="javascript:', rendered.lower())
        self.assertIsNone(re.search(r"<[^>]+\sonerror\s*=", rendered, flags=re.IGNORECASE))
        self.assertIn('&lt;script&gt;alert(&quot;tweet&quot;)&lt;/script&gt;<br>', rendered)
        self.assertIn('&lt;script&gt;alert(&quot;llm&quot;)&lt;/script&gt;', rendered)
        self.assertIn('&lt;img src=x onerror=alert(1)&gt;', rendered)
        self.assertIn('正常中文 and English', rendered)
        self.assertIn('href="https://example.com/article?a=1&amp;b=2"', rendered)
        self.assertIn('https://x.com/alice%22%20onmouseover%3D%22alert%281%29/status/123', rendered)
        self.assertIn('&lt;b onclick=&quot;alert(1)&quot;&gt;Model&lt;/b&gt;', rendered)


if __name__ == "__main__":
    unittest.main()
