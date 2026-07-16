import unittest

from app.security import (
    escape_html_text,
    mask_secret,
    redact_sensitive_text,
    safe_html_attribute,
    safe_image_src_attribute,
    safe_url_attribute,
    sanitize_external_url,
    sanitize_image_src,
)


class HtmlSafetyTests(unittest.TestCase):
    def test_escape_html_text_blocks_script_and_preserves_newlines(self):
        value = '<script>alert("x")</script>\r\n中文与 English'
        escaped = escape_html_text(value)

        self.assertEqual(
            escaped,
            '&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;<br>\n中文与 English',
        )
        self.assertNotIn("<script", escaped)

    def test_safe_html_attribute_neutralizes_quote_and_onerror_payload(self):
        payload = 'photo.jpg" onerror="alert(1)'
        escaped = safe_html_attribute(payload)

        self.assertEqual(escaped, 'photo.jpg&quot; onerror=&quot;alert(1)')
        self.assertNotIn('"', escaped)

    def test_safe_html_attribute_handles_none_and_control_characters(self):
        self.assertEqual(safe_html_attribute(None), "")
        self.assertEqual(safe_html_attribute("中\x00文\nEnglish"), "中 文 English")


class UrlSafetyTests(unittest.TestCase):
    def test_accepts_normal_x_and_https_urls(self):
        urls = (
            "https://x.com/openai/status/1234567890",
            "https://example.com/news?id=42#summary",
            "http://localhost:11434/api/tags",
        )
        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(sanitize_external_url(url), url)

    def test_only_http_and_https_schemes_are_allowed(self):
        dangerous = (
            "javascript:alert(1)",
            "JaVaScRiPt:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "data:image/svg+xml,<svg onload=alert(1)>",
            "file:///etc/passwd",
            "vbscript:msgbox(1)",
            "//evil.example/path",
        )
        for url in dangerous:
            with self.subTest(url=url):
                self.assertEqual(sanitize_external_url(url), "")

    def test_rejects_attribute_and_parser_smuggling(self):
        dangerous = (
            'https://example.com/image.jpg" onerror="alert(1)',
            "https://example.com/<script>alert(1)</script>",
            "https://example.com\\@evil.example/",
            "https://user:password@example.com/",
            "https://example.com/line\nbreak",
        )
        for url in dangerous:
            with self.subTest(url=url):
                self.assertEqual(sanitize_external_url(url, fallback="#"), "#")

    def test_safe_url_attribute_validates_before_escaping(self):
        self.assertEqual(safe_url_attribute("javascript:alert(1)", fallback="#"), "#")
        self.assertEqual(
            safe_url_attribute("https://example.com/?a=1&b=2"),
            "https://example.com/?a=1&amp;b=2",
        )

    def test_image_sources_reject_data_by_default(self):
        png_data = "data:image/png;base64,iVBORw0KGgo="
        self.assertEqual(sanitize_image_src(png_data), "")
        self.assertEqual(sanitize_image_src(png_data, allow_data_image=True), png_data)
        self.assertEqual(
            safe_image_src_attribute(png_data, allow_data_image=True),
            png_data,
        )
        self.assertEqual(
            sanitize_image_src("data:image/svg+xml;base64,PHN2Zz4=", allow_data_image=True),
            "",
        )


class SecretMaskingTests(unittest.TestCase):
    def test_masks_empty_and_short_secrets(self):
        self.assertEqual(mask_secret(None), "")
        self.assertEqual(mask_secret("abc123"), "******")

    def test_masks_long_secret_but_keeps_small_identifier(self):
        secret = "sk-live-super-secret-9Z"
        masked = mask_secret(secret)

        self.assertEqual(len(masked), len(secret))
        self.assertTrue(masked.startswith("sk-"))
        self.assertTrue(masked.endswith("9Z"))
        self.assertNotIn("live-super-secret", masked)
        self.assertNotEqual(masked, secret)

    def test_masking_options_are_validated(self):
        with self.assertRaises(ValueError):
            mask_secret("secret", visible_prefix=-1)
        with self.assertRaises(ValueError):
            mask_secret("secret", mask_character="**")

    def test_redacts_explicit_bearer_and_sk_style_credentials(self):
        explicit = "custom-secret-value"
        bearer = "eyJhbGciOiJIUzI1NiJ9.payload.signature"
        sk_token = "sk-proj-1234567890abcdef"
        message = (
            f"explicit={explicit}; Authorization: Bearer {bearer}; "
            f"provider rejected {sk_token}"
        )

        redacted = redact_sensitive_text(message, secrets=(explicit,))

        self.assertNotIn(explicit, redacted)
        self.assertNotIn(bearer, redacted)
        self.assertNotIn(sk_token, redacted)
        self.assertIn("Bearer [REDACTED]", redacted)
        self.assertGreaterEqual(redacted.count("[REDACTED]"), 3)

    def test_redaction_leaves_short_non_credentials_alone(self):
        self.assertEqual(
            redact_sensitive_text("use sk-model with Bearer note"),
            "use sk-model with Bearer note",
        )


if __name__ == "__main__":
    unittest.main()
