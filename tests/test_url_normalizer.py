from __future__ import annotations

import socket
import unittest

from app.network_security import NetworkSecurityError, NetworkSecurityValidator
from app.url_normalizer import UrlNormalizationError, UrlNormalizer


PUBLIC_IP = "93.184.216.34"


class UrlNormalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = UrlNormalizer()

    def test_normal_https_url_has_stable_sha256_identity(self) -> None:
        value = self.normalizer.normalize("https://example.com/article")
        self.assertEqual("https://example.com/article", value.normalized_url)
        self.assertEqual("example.com", value.domain)
        self.assertEqual(64, len(value.url_hash))
        self.assertEqual(value.url_hash, self.normalizer.normalize(value.original_url).url_hash)

    def test_scheme_hostname_default_port_fragment_and_empty_path_are_normalized(self) -> None:
        value = self.normalizer.normalize("HTTPS://EXAMPLE.COM:443#section")
        self.assertEqual("https://example.com/", value.normalized_url)

    def test_non_default_port_is_retained(self) -> None:
        self.assertEqual(
            "https://example.com:8443/story",
            self.normalizer.normalize("https://example.com:8443/story").normalized_url,
        )

    def test_explicit_tracking_parameters_are_removed_and_recorded(self) -> None:
        value = self.normalizer.normalize(
            "https://example.com/story?id=7&utm_source=x&fbclid=abc&utm_campaign=test"
        )
        self.assertEqual("https://example.com/story?id=7", value.normalized_url)
        self.assertEqual(
            ["utm_source", "fbclid", "utm_campaign"],
            value.removed_tracking_params,
        )

    def test_unknown_and_content_queries_preserve_order_and_encoding(self) -> None:
        url = "https://example.com/story?lang=zh-CN&id=42&page=2&q=a%20b"
        self.assertEqual(url, self.normalizer.normalize(url).normalized_url)

    def test_unicode_hostname_is_stored_as_punycode(self) -> None:
        value = self.normalizer.normalize("https://例子.测试/文章")
        self.assertTrue(value.domain.startswith("xn--"))
        self.assertIn("/文章", value.normalized_url)

    def test_malformed_percent_encoding_is_rejected(self) -> None:
        with self.assertRaisesRegex(UrlNormalizationError, "percent"):
            self.normalizer.normalize("https://example.com/bad%2")

    def test_embedded_credentials_are_rejected(self) -> None:
        with self.assertRaisesRegex(UrlNormalizationError, "Credentials"):
            self.normalizer.normalize("https://user:password@example.com/story")

    def test_unsupported_and_confusing_urls_are_rejected(self) -> None:
        for value in (
            "javascript:alert(1)",
            "file:///etc/passwd",
            "ftp://example.com/file",
            "https:\\example.com\\story",
            "https://example.com/a b",
        ):
            with self.subTest(value=value), self.assertRaises(UrlNormalizationError):
                self.normalizer.normalize(value)

    def test_obvious_private_and_metadata_targets_are_rejected(self) -> None:
        for value in (
            "http://localhost/story",
            "http://127.0.0.1/story",
            "http://10.0.0.1/story",
            "http://[::1]/story",
            "http://169.254.169.254/latest/meta-data",
            "http://100.100.100.200/latest/meta-data",
            "http://metadata.google.internal/",
        ):
            with self.subTest(value=value), self.assertRaises(UrlNormalizationError) as caught:
                self.normalizer.normalize(value)
            self.assertEqual("blocked_private_address", caught.exception.code)

    def test_safe_related_canonical_takes_priority(self) -> None:
        value = self.normalizer.normalize(
            "https://www.example.com/story?utm_source=x",
            resolved_url="https://www.example.com/story?id=9",
            canonical_url="https://example.com/articles/9",
        )
        self.assertEqual("https://example.com/articles/9", value.normalized_url)

    def test_unrelated_canonical_is_preserved_but_not_selected(self) -> None:
        value = self.normalizer.normalize(
            "https://example.com/story",
            resolved_url="https://example.com/story/7",
            canonical_url="https://unrelated.test/other",
        )
        self.assertEqual("https://example.com/story/7", value.normalized_url)
        self.assertEqual("https://unrelated.test/other", value.canonical_url)

    def test_signed_url_is_not_rewritten_or_stripped(self) -> None:
        url = (
            "https://cdn.example.com/file?X-Amz-Signature=ABC&"
            "utm_source=required-by-signature&id=7"
        )
        value = self.normalizer.normalize(url)
        self.assertEqual(url, value.normalized_url)
        self.assertEqual([], value.removed_tracking_params)

    def test_content_identity_queries_languages_versions_and_paths_do_not_merge(self) -> None:
        pairs = [
            ("https://example.com/story?id=1", "https://example.com/story?id=2"),
            ("https://example.com/story?lang=en", "https://example.com/story?lang=zh"),
            ("https://example.com/story?v=1", "https://example.com/story?v=2"),
            ("https://example.com/a", "https://example.com/b"),
            ("https://example.com/search?q=agent", "https://example.com/story/agent"),
        ]
        for left, right in pairs:
            with self.subTest(left=left, right=right):
                self.assertNotEqual(
                    self.normalizer.normalize(left).url_hash,
                    self.normalizer.normalize(right).url_hash,
                )


class NetworkSecurityTests(unittest.IsolatedAsyncioTestCase):
    async def test_public_dns_target_is_allowed_and_all_addresses_retained(self) -> None:
        validator = NetworkSecurityValidator(
            resolver=lambda _host, _port: [PUBLIC_IP, "8.8.8.8"]
        )
        target = await validator.validate_url("https://example.com/story")
        self.assertEqual((PUBLIC_IP, "8.8.8.8"), target.addresses)

    async def test_dns_resolving_to_any_private_address_is_blocked(self) -> None:
        validator = NetworkSecurityValidator(
            resolver=lambda _host, _port: [PUBLIC_IP, "192.168.1.10"]
        )
        with self.assertRaises(NetworkSecurityError) as caught:
            await validator.validate_url("https://example.com/story")
        self.assertEqual("blocked_private_address", caught.exception.code)

    async def test_dns_failure_is_classified(self) -> None:
        def fail(_host, _port):
            raise socket.gaierror("TEST DNS failure")

        with self.assertRaises(NetworkSecurityError) as caught:
            await NetworkSecurityValidator(resolver=fail).validate_url(
                "https://example.com/story"
            )
        self.assertEqual("dns_failure", caught.exception.code)

    async def test_internal_single_label_and_local_domains_are_blocked_without_dns(self) -> None:
        calls = []
        validator = NetworkSecurityValidator(
            resolver=lambda host, port: calls.append((host, port)) or [PUBLIC_IP]
        )
        for value in ("http://intranet/", "http://printer.local/"):
            with self.subTest(value=value), self.assertRaises(NetworkSecurityError):
                await validator.validate_url(value)
        self.assertEqual([], calls)


if __name__ == "__main__":
    unittest.main()
