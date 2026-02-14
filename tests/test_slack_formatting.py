"""Tests for Slack message formatting and posting."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from scripts.digest import (
    LinearTicket,
    SLACK_BLOCK_TEXT_LIMIT,
    format_for_slack,
    post_to_slack,
    chunk_slack_text,
    trim_message,
)


def _ticket(identifier: str, url: str) -> LinearTicket:
    return LinearTicket(
        identifier=identifier,
        title="Some ticket",
        url=url,
        state="Done",
        team="Team",
        project=None,
        found=True,
    )


GH_ORG = "ledger-rocket"


class TestFormatForSlack:
    def test_converts_markdown_headings_to_bold(self):
        narrative = "## ✨ Features\nSome text\n### 🐛 Fixes\nMore text"
        result = format_for_slack(narrative, {})
        assert "*✨ Features*" in result
        assert "*🐛 Fixes*" in result
        assert "##" not in result

    def test_single_hash_heading(self):
        narrative = "# Big Heading\nContent"
        result = format_for_slack(narrative, {})
        assert "*Big Heading*" in result
        assert "#" not in result.replace("*Big Heading*", "")

    def test_replaces_ticket_ids_with_linear_links(self):
        tickets = {
            "PRO-1113": _ticket(
                "PRO-1113",
                "https://linear.app/team/issue/PRO-1113/some-title",
            ),
        }
        narrative = "Fixed a bug in PRO-1113 that caused issues."
        result = format_for_slack(narrative, tickets)
        assert "<https://linear.app/team/issue/PRO-1113/some-title|PRO-1113>" in result
        assert "PRO-1113" not in result.replace(
            "<https://linear.app/team/issue/PRO-1113/some-title|PRO-1113>", ""
        )

    def test_skips_tickets_without_url(self):
        tickets = {
            "PRO-999": _ticket("PRO-999", ""),
        }
        narrative = "Something about PRO-999."
        result = format_for_slack(narrative, tickets)
        assert "<|PRO-999>" not in result
        assert "PRO-999" in result

    def test_replaces_pr_refs_with_github_links(self):
        narrative = "dashboard#240 and api-server#55"
        result = format_for_slack(narrative, {}, gh_org=GH_ORG)
        assert f"<https://github.com/{GH_ORG}/dashboard/pull/240|dashboard#240>" in result
        assert f"<https://github.com/{GH_ORG}/api-server/pull/55|api-server#55>" in result

    def test_pr_refs_without_org_are_unchanged(self):
        narrative = "dashboard#240"
        result = format_for_slack(narrative, {})
        assert "dashboard#240" in result
        assert "<https://" not in result

    def test_strips_org_prefix_from_pr_refs(self):
        narrative = "ledger-rocket/dashboard#240"
        result = format_for_slack(narrative, {}, gh_org=GH_ORG)
        assert f"<https://github.com/{GH_ORG}/dashboard/pull/240|dashboard#240>" in result
        assert "ledger-rocket/" not in result.replace(
            f"https://github.com/{GH_ORG}/", ""
        )

    def test_renders_mentions_as_bold(self):
        narrative = "@stepansin and @laurencehook-lr"
        result = format_for_slack(narrative, {})
        assert "*@stepansin*" in result
        assert "*@laurencehook-lr*" in result
        assert "<https://github.com/" not in result

    def test_combined_formatting(self):
        tickets = {
            "PRO-1124": _ticket(
                "PRO-1124",
                "https://linear.app/team/issue/PRO-1124/feature",
            ),
        }
        narrative = (
            "## ✨ Features\n"
            "Added a new flow for account creation.\n"
            "[PRO-1124; dashboard#240 @laurencehook-lr]\n"
        )
        result = format_for_slack(narrative, tickets, gh_org=GH_ORG)
        assert "*✨ Features*" in result
        assert "<https://linear.app/team/issue/PRO-1124/feature|PRO-1124>" in result
        assert f"<https://github.com/{GH_ORG}/dashboard/pull/240|dashboard#240>" in result
        assert "*@laurencehook-lr*" in result
        assert "##" not in result

    def test_no_changes_when_already_formatted(self):
        narrative = "*✨ Features*\nSome text with no tickets or mentions."
        result = format_for_slack(narrative, {})
        assert result == narrative


class TestChunkSlackText:
    def test_no_link_split_across_chunks(self):
        """A Slack hyperlink must never be split across two chunks."""
        link = "<https://github.com/ledger-rocket/dashboard/pull/240|dashboard#240>"
        # Build a message where the link sits right at the boundary
        padding = "x" * (SLACK_BLOCK_TEXT_LIMIT - 10) + "\n" + link
        chunks = chunk_slack_text(padding)
        for chunk in chunks:
            assert chunk.count("<") == chunk.count(">"), (
                f"Unbalanced angle brackets in chunk: {chunk[-120:]}"
            )

    def test_empty_message(self):
        assert chunk_slack_text("") == [""]

    def test_short_message_single_chunk(self):
        msg = "Hello\nWorld"
        assert chunk_slack_text(msg) == ["Hello\nWorld"]


class TestPostToSlack:
    def test_slack_payload_structure(self):
        """Verify the shape of the JSON sent to Slack."""
        with patch("scripts.digest.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            message = "*✨ Features*\nSome feature was added.\n[PRO-123; repo#1 *@dev*]"
            window_text = "Release window: 2026-02-09 05:37 UTC → 2026-02-10 05:37 UTC"

            post_to_slack("https://hooks.slack.com/test", message, window_text)

            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")

            assert "text" in payload
            assert "blocks" in payload

            blocks = payload["blocks"]
            assert blocks[0]["type"] == "header"
            assert blocks[1]["type"] == "context"
            assert blocks[2]["type"] == "divider"
            # Content sections follow
            assert len(blocks) > 3
            for block in blocks[3:]:
                assert block["type"] == "section"
                assert block["text"]["type"] == "mrkdwn"

    def test_slack_payload_contains_formatted_message(self):
        """Verify that the Slack payload text contains our formatted content."""
        with patch("scripts.digest.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            message = (
                "*✨ Features*\n"
                "Added account creation flow.\n"
                "[<https://linear.app/t/PRO-1124|PRO-1124>; "
                f"<https://github.com/{GH_ORG}/dashboard/pull/240|dashboard#240> "
                "*@dev*]"
            )
            post_to_slack("https://hooks.slack.com/test", message, "window")

            payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
            section_texts = [
                b["text"]["text"] for b in payload["blocks"] if b["type"] == "section"
            ]
            full_text = "\n".join(section_texts)
            assert "*✨ Features*" in full_text
            assert "<https://linear.app/t/PRO-1124|PRO-1124>" in full_text
            assert "*@dev*" in full_text
