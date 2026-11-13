"""YouTube Music links are searched from metadata and never fabricated."""

import unittest
from urllib.parse import parse_qs, urlsplit

from backend.recommender.external_links import listening_links, verified_youtube_music_url, youtube_music_search_url


class ExternalListeningLinkTests(unittest.TestCase):
    def test_search_url_encodes_title_artist_and_unicode(self):
        url = youtube_music_search_url("Don't Stop, 東京 & Me", "Zoë O'Neil")
        parsed = urlsplit(url)
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.netloc, "music.youtube.com")
        self.assertEqual(parse_qs(parsed.query)["q"], ["Don't Stop, 東京 & Me Zoë O'Neil"])
        self.assertIn("%26", parsed.query)

    def test_verified_direct_link_is_preferred_and_invalid_url_is_rejected(self):
        exact = "https://music.youtube.com/watch?v=known-id123"
        self.assertEqual(verified_youtube_music_url(exact), exact)
        self.assertIsNone(verified_youtube_music_url("https://music.youtube.com/watch?v=short"))
        self.assertIsNone(verified_youtube_music_url("https://example.com/watch?v=known-id123"))
        self.assertIsNone(verified_youtube_music_url("javascript:alert(1)"))
        links = listening_links("Song", "Artist", exact)
        self.assertEqual(links["youtube_music_url"], exact)
        self.assertIsNotNone(links["youtube_music_search_url"])

    def test_missing_title_and_artist_has_no_search_url(self):
        self.assertIsNone(youtube_music_search_url(None, "  "))


if __name__ == "__main__":
    unittest.main()
