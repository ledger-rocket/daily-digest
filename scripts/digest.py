#!/usr/bin/env python3
"""Generate a daily narrative digest from merged PRs and Linear tickets."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import textwrap
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import cast

import requests
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI, RateLimitError
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"
DEFAULT_TICKET_REGEX = r"([A-Z][A-Z0-9]+-\d+)"
DEFAULT_HOURS = 24
DEFAULT_OPENAI_MODEL = "gpt-5.2"
MAX_BODY_CHARS = 4000
MAX_SLACK_CHARS = 39000
SLACK_BLOCK_TEXT_LIMIT = 3000
SLACK_MAX_BLOCKS = 50
FLEX_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.2": (0.875, 7.00),
    "gpt-5.1": (0.625, 5.00),
    "gpt-5": (0.625, 5.00),
    "gpt-5-mini": (0.125, 1.00),
    "gpt-5-nano": (0.025, 0.20),
    "o3": (1.00, 4.00),
    "o4-mini": (0.55, 0.138),
}

PR_SUMMARY_SYSTEM_PROMPT = (
    "You summarize a single PR. Only use facts provided. "
    "Do not mention tests, CI, or lint results. "
    "Use 1-2 short sentences. No speculation."
)
NARRATIVE_SYSTEM_PROMPT = (
    "You are a concise engineering release narrator. Only use facts provided. "
    "Do not invent details. Do not mention tests, CI, or lint results. "
    "Use friendly, readable prose with 1-3 short sentences per ticket. "
    "Group by sections with emoji headings: '*✨ Features*', '*🐛 Fixes*', "
    "'*🧰 Infra / Refactor / Chore*'. "
    "Do not include a release window line; it is provided separately. "
    "After each paragraph, add a bracketed line with references in this format: "
    "[TICKET-123; repo#123 @author, repo#456 @author]. "
    "If multiple PRs share a ticket, list them all in the bracketed line. "
    "Keep language tight and functional."
)
PR_SUMMARY_USER_PROMPT_PREFIX = "Summarize this PR using the data provided.\n\nDATA:\n"
NARRATIVE_USER_PROMPT_PREFIX = (
    "Summarize the following merged PRs and linked Linear tickets into a daily "
    "narrative. If a ticket is unknown, do not guess its details.\n\nDATA:\n"
)
JSONDict = dict[str, object]
JSONList = list[object]
HTTP_ERROR_THRESHOLD = 400
OPENAI_MAX_ATTEMPTS = 5
TICKET_PARTS = 2
MISSING_GITHUB_AUTH_MESSAGE = "Missing GitHub API credentials."


def ensure_dict(value: object, _context: str) -> JSONDict:
    """Return a dictionary value or raise."""
    if isinstance(value, dict):
        return cast("JSONDict", value)
    raise TypeError


def ensure_list(value: object, _context: str) -> JSONList:
    """Return a list value or raise."""
    if isinstance(value, list):
        return cast("JSONList", value)
    raise TypeError


def ensure_str(value: object, _context: str, default: str = "") -> str:
    """Return a string value or a default."""
    if isinstance(value, str):
        return value
    if value is None:
        return default
    raise TypeError


def ensure_int(value: object, _context: str) -> int:
    """Return an integer value or raise."""
    if isinstance(value, int):
        return value
    raise TypeError


class GraphQLRequestError(RuntimeError):
    """Raised when a GraphQL request fails."""

    def __init__(self, status_code: int, text: str) -> None:
        """Create a GraphQL request error."""
        super().__init__(f"GraphQL request failed ({status_code}): {text}")


class SlackWebhookError(RuntimeError):
    """Raised when a Slack webhook call fails."""

    def __init__(self, status_code: int, text: str) -> None:
        """Create a Slack webhook error."""
        super().__init__(f"Slack webhook failed ({status_code}): {text}")


class OpenAIRetryError(RuntimeError):
    """Raised when OpenAI retries are exhausted."""


class GraphQLErrorsError(RuntimeError):
    """Raised when GraphQL response includes errors."""

    def __init__(self, errors: object) -> None:
        """Create a GraphQL errors exception."""
        super().__init__(f"GraphQL errors: {errors}")


class OpenAIEmptyResponseError(RuntimeError):
    """Raised when OpenAI returns no content."""

    def __init__(self) -> None:
        """Create an empty response error."""
        super().__init__("OpenAI response missing content.")


class Settings(BaseSettings):
    """Environment-backed settings for the digest."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    digest_gh_org: str = Field(alias="DIGEST_GH_ORG")
    digest_repos: str | None = Field(default=None, alias="DIGEST_REPOS")
    digest_hours: int = Field(default=DEFAULT_HOURS, alias="DIGEST_HOURS")
    digest_include_unlinked: bool = Field(default=True, alias="DIGEST_INCLUDE_UNLINKED")
    digest_ticket_regex: str = Field(
        default=DEFAULT_TICKET_REGEX,
        alias="DIGEST_TICKET_REGEX",
    )
    digest_ticket_regex_ignorecase: bool = Field(
        default=True,
        alias="DIGEST_TICKET_REGEX_IGNORECASE",
    )

    github_token: str | None = Field(default=None, alias="GITHUB_TOKEN")
    gh_token: str | None = Field(default=None, alias="GH_TOKEN")
    digest_gh_token: str | None = Field(default=None, alias="DIGEST_GH_TOKEN")
    linear_api_key: str = Field(alias="LINEAR_API_KEY")
    slack_webhook_url: str | None = Field(default=None, alias="SLACK_WEBHOOK_URL")

    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    openai_model: str = Field(default=DEFAULT_OPENAI_MODEL, alias="OPENAI_MODEL")
    openai_price_input_per_million: float | None = Field(
        default=None,
        alias="OPENAI_PRICE_INPUT_PER_MILLION",
    )
    openai_price_output_per_million: float | None = Field(
        default=None,
        alias="OPENAI_PRICE_OUTPUT_PER_MILLION",
    )


def get_settings() -> Settings:
    """Load settings from environment variables."""
    return Settings.model_validate({})


def make_unknown_ticket(identifier: str) -> LinearTicket:
    """Return a placeholder ticket for unknown identifiers."""
    return LinearTicket(
        identifier=identifier,
        title="Unknown ticket",
        url="",
        state="Unknown",
        team=None,
        project=None,
        found=False,
    )


def log_elapsed(message: str, start: float, **fields: object) -> None:
    """Log elapsed time with additional fields."""
    elapsed = f"{time.perf_counter() - start:.2f}s"
    logger.info(
        "{message} (elapsed {elapsed})",
        message=message,
        elapsed=elapsed,
        **fields,
    )


class CommitInfo(BaseModel):
    """Minimal commit info for summarization."""

    headline: str
    body: str


class PullRequest(BaseModel):
    """Merged PR data used for digest generation."""

    repo: str
    number: int
    url: str
    title: str
    body: str
    merged_at: str
    author: str
    labels: list[str]
    tickets: list[str]
    commits: list[CommitInfo]
    summary: str | None = None


class LinearTicket(BaseModel):
    """Linear ticket data resolved from identifiers."""

    identifier: str
    title: str
    url: str
    state: str
    team: str | None
    project: str | None
    found: bool


class UsageTotals(BaseModel):
    """Accumulates OpenAI usage totals."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class UsageTracker(BaseModel):
    """Tracks OpenAI usage totals and optional cost estimates."""

    input_per_million: float | None = None
    output_per_million: float | None = None
    totals: UsageTotals = Field(default_factory=UsageTotals)

    def add(self, usage: object) -> None:
        """Add usage to totals."""
        prompt_tokens = usage_int(usage, "prompt_tokens") or 0
        completion_tokens = usage_int(usage, "completion_tokens") or 0
        total_tokens = usage_int(usage, "total_tokens")
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        self.totals.prompt_tokens += prompt_tokens
        self.totals.completion_tokens += completion_tokens
        self.totals.total_tokens += total_tokens
        self.totals.cost_usd += estimate_cost(
            prompt_tokens,
            completion_tokens,
            self.input_per_million,
            self.output_per_million,
        )

    def to_log(self) -> dict[str, object]:
        """Return a log-friendly usage summary."""
        payload: dict[str, object] = {
            "prompt_tokens": self.totals.prompt_tokens,
            "completion_tokens": self.totals.completion_tokens,
            "total_tokens": self.totals.total_tokens,
        }
        if self.input_per_million is None or self.output_per_million is None:
            payload["cost_usd"] = None
            payload["pricing"] = "unset"
            return payload
        payload["cost_usd"] = round(self.totals.cost_usd, 6)
        payload["pricing"] = {
            "input_per_million": self.input_per_million,
            "output_per_million": self.output_per_million,
        }
        return payload


def usage_int(usage: object, field: str) -> int | None:
    """Extract an integer usage field."""
    value: object | None
    if isinstance(usage, dict):
        value = usage.get(field)
    else:
        value = getattr(usage, field, None)
    if isinstance(value, int):
        return value
    return None


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_per_million: float | None,
    output_per_million: float | None,
) -> float:
    """Estimate cost from token counts."""
    cost = 0.0
    if input_per_million is not None:
        cost += (prompt_tokens / 1_000_000) * input_per_million
    if output_per_million is not None:
        cost += (completion_tokens / 1_000_000) * output_per_million
    return cost


def resolve_pricing(settings: Settings) -> tuple[float | None, float | None]:
    """Resolve pricing from env overrides or Flex defaults."""
    if (
        settings.openai_price_input_per_million is not None
        and settings.openai_price_output_per_million is not None
    ):
        return (
            settings.openai_price_input_per_million,
            settings.openai_price_output_per_million,
        )
    pricing = FLEX_PRICING.get(settings.openai_model)
    if pricing is None:
        return None, None
    return pricing


class DigestItem(BaseModel):
    """Grouped PRs under a ticket with a category."""

    ticket_id: str | None
    ticket: LinearTicket | None
    prs: list[PullRequest]
    category: str


def iso_window(hours_back: int) -> tuple[str, str, str, str]:
    """Return UTC start/end timestamps and display strings."""
    now = datetime.now(UTC)
    start = now - timedelta(hours=hours_back)
    return (
        start.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        start.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        now.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )


def github_headers(settings: Settings) -> dict[str, str]:
    """Return GitHub API headers with authentication."""
    token = settings.github_token or settings.gh_token or settings.digest_gh_token
    if not token:
        raise SystemExit(MISSING_GITHUB_AUTH_MESSAGE)
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def linear_headers(settings: Settings) -> dict[str, str]:
    """Return Linear API headers with authentication."""
    return {"Authorization": settings.linear_api_key, "Accept": "application/json"}


def call_graphql(
    url: str,
    headers: dict[str, str],
    query: str,
    variables: JSONDict,
) -> JSONDict:
    """Call a GraphQL endpoint and return the data payload."""
    response = requests.post(
        url,
        headers=headers,
        json={"query": query, "variables": variables},
        timeout=30,
    )
    if response.status_code >= HTTP_ERROR_THRESHOLD:
        raise GraphQLRequestError(response.status_code, response.text)
    payload = ensure_dict(response.json(), "GraphQL response")
    errors = payload.get("errors")
    if errors:
        raise GraphQLErrorsError(errors)
    return ensure_dict(payload.get("data"), "GraphQL data")


def build_search_query(org: str, since_iso: str, repos: list[str] | None) -> list[str]:
    """Build GitHub search queries for merged PRs."""
    base = f"is:pr is:merged merged:>={since_iso}"
    if repos:
        return [f"repo:{org}/{repo} {base}" for repo in repos]
    return [f"org:{org} {base}"]


def extract_pr_labels(node_dict: JSONDict) -> list[str]:
    """Extract label names from a PR node."""
    labels_container = ensure_dict(node_dict.get("labels") or {}, "labels")
    label_nodes = ensure_list(labels_container.get("nodes") or [], "labels.nodes")
    labels: list[str] = []
    for label in label_nodes:
        label_dict = ensure_dict(label, "label")
        label_name = ensure_str(label_dict.get("name"), "label.name")
        if label_name:
            labels.append(label_name)
    return labels


def extract_pr_commits(node_dict: JSONDict) -> list[CommitInfo]:
    """Extract commit info from a PR node."""
    commits: list[CommitInfo] = []
    commits_container = ensure_dict(node_dict.get("commits") or {}, "commits")
    commit_nodes = ensure_list(commits_container.get("nodes") or [], "commit.nodes")
    for commit_node in commit_nodes:
        commit_node_dict = ensure_dict(commit_node, "commit_node")
        commit = ensure_dict(commit_node_dict.get("commit") or {}, "commit")
        headline = ensure_str(
            commit.get("messageHeadline"),
            "commit.messageHeadline",
        ).strip()
        message_body = ensure_str(
            commit.get("messageBody"),
            "commit.messageBody",
        ).strip()
        if headline:
            commits.append(CommitInfo(headline=headline, body=message_body))
    return commits


def extract_pr_author(node_dict: JSONDict) -> str:
    """Extract PR author login."""
    author_dict = ensure_dict(node_dict.get("author") or {}, "author")
    return ensure_str(author_dict.get("login"), "author.login", "unknown")


def extract_repo_name(node_dict: JSONDict) -> str:
    """Extract repository name for a PR."""
    repository = ensure_dict(node_dict.get("repository"), "repository")
    return ensure_str(repository.get("name"), "repository.name")


def parse_pr_node(node_dict: JSONDict) -> PullRequest:
    """Parse a PullRequest model from a GraphQL node."""
    body = ensure_str(node_dict.get("body"), "body")
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + "..."
    return PullRequest(
        repo=extract_repo_name(node_dict),
        number=ensure_int(node_dict.get("number"), "number"),
        url=ensure_str(node_dict.get("url"), "url"),
        title=ensure_str(node_dict.get("title"), "title"),
        body=body,
        merged_at=ensure_str(node_dict.get("mergedAt"), "mergedAt"),
        author=extract_pr_author(node_dict),
        labels=extract_pr_labels(node_dict),
        tickets=[],
        commits=extract_pr_commits(node_dict),
    )


def fetch_merged_prs(
    settings: Settings,
    org: str,
    since_iso: str,
    repos: list[str] | None,
) -> list[PullRequest]:
    """Fetch merged PRs from GitHub."""
    query = textwrap.dedent(
        """
        query($query: String!, $cursor: String) {
          search(type: ISSUE, query: $query, first: 100, after: $cursor) {
            pageInfo { hasNextPage endCursor }
            nodes {
              ... on PullRequest {
                title
                body
                url
                number
                mergedAt
                author { login }
                labels(first: 30) { nodes { name } }
                commits(first: 50) {
                  nodes {
                    commit {
                      messageHeadline
                      messageBody
                    }
                  }
                }
                repository { name }
              }
            }
          }
        }
        """,
    ).strip()

    all_prs: list[PullRequest] = []
    for search_query in build_search_query(org, since_iso, repos):
        logger.info("GitHub search query: {query}", query=search_query)
        cursor: str | None = None
        while True:
            data = call_graphql(
                GITHUB_GRAPHQL_URL,
                github_headers(settings),
                query,
                {"query": search_query, "cursor": cursor},
            )
            search = ensure_dict(data.get("search"), "search")
            nodes = ensure_list(search.get("nodes") or [], "search.nodes")
            logger.info("Retrieved PR page: {count}", count=len(nodes))
            for node in nodes:
                node_dict = ensure_dict(node, "pull_request")
                all_prs.append(parse_pr_node(node_dict))
            page_info = ensure_dict(search.get("pageInfo"), "search.pageInfo")
            if not bool(page_info.get("hasNextPage")):
                break
            end_cursor = page_info.get("endCursor")
            cursor = ensure_str(end_cursor, "pageInfo.endCursor", "") or None
    return all_prs


def normalize_ticket(raw_ticket: str) -> str | None:
    """Normalize a ticket identifier or return None."""
    parts = raw_ticket.split("-", 1)
    if len(parts) != TICKET_PARTS:
        return None
    team_key, number_text = parts[0].upper(), parts[1]
    try:
        int(number_text)
    except ValueError:
        return None
    return f"{team_key}-{number_text}"


def extract_tickets(pr: PullRequest, regex: re.Pattern[str]) -> list[str]:
    """Extract normalized ticket identifiers from a PR."""
    text = f"{pr.title}\n{pr.body}"
    normalized: list[str] = []
    for match in regex.findall(text):
        ticket = normalize_ticket(match)
        if ticket:
            normalized.append(ticket)
    return sorted(set(normalized))


def fetch_linear_ticket(settings: Settings, identifier: str) -> LinearTicket:
    """Fetch a Linear ticket by identifier."""
    if "-" not in identifier:
        return make_unknown_ticket(identifier)
    team_key, number_text = identifier.split("-", 1)
    try:
        number = int(number_text)
    except ValueError:
        return make_unknown_ticket(identifier)

    query = textwrap.dedent(
        """
        query($team: String!, $number: Float!) {
          issues(
            filter: { team: { key: { eq: $team } }, number: { eq: $number } },
            first: 1
          ) {
            nodes {
              identifier
              title
              url
              state { name }
              team { name }
              project { name }
            }
          }
        }
        """,
    ).strip()

    data = call_graphql(
        LINEAR_GRAPHQL_URL,
        linear_headers(settings),
        query,
        {"team": team_key, "number": number},
    )
    issues = ensure_dict(data.get("issues"), "issues")
    nodes = ensure_list(issues.get("nodes") or [], "issues.nodes")
    issue = ensure_dict(nodes[0], "issue") if nodes else None
    if not issue:
        return make_unknown_ticket(identifier)
    return LinearTicket(
        identifier=ensure_str(issue.get("identifier"), "issue.identifier", identifier),
        title=ensure_str(issue.get("title"), "issue.title", "Unknown title"),
        url=ensure_str(issue.get("url"), "issue.url", ""),
        state=ensure_str(
            ensure_dict(issue.get("state") or {}, "issue.state").get("name"),
            "state.name",
            "",
        ),
        team=ensure_str(
            ensure_dict(issue.get("team") or {}, "issue.team").get("name"),
            "team.name",
            "",
        )
        or None,
        project=ensure_str(
            ensure_dict(issue.get("project") or {}, "issue.project").get("name"),
            "project.name",
            "",
        )
        or None,
        found=True,
    )


def categorize_pr(pr: PullRequest) -> str:
    """Categorize a PR using conventional commit prefixes."""
    title = pr.title.strip().lower()
    if title.startswith(("feat(", "feat:")):
        return "Features"
    if title.startswith(("fix(", "fix:")):
        return "Fixes"
    if title.startswith(("refactor(", "refactor:")):
        return "Infra/Refactor/Chore"
    if title.startswith(("chore(", "chore:")):
        return "Infra/Refactor/Chore"
    return "Infra/Refactor/Chore"


def category_priority(category: str) -> int:
    """Return priority for category ordering."""
    if category == "Features":
        return 3
    if category == "Fixes":
        return 2
    return 1


def pick_category(categories: list[str]) -> str:
    """Pick the highest priority category."""
    if categories:
        return max(categories, key=category_priority)
    return "Infra/Refactor/Chore"


def group_digest_items(
    *,
    prs: list[PullRequest],
    include_unlinked: bool,
) -> tuple[list[DigestItem], list[PullRequest]]:
    """Group PRs by ticket and return grouped items plus unlinked PRs."""
    ticket_groups: dict[str, list[PullRequest]] = defaultdict(list)
    unlinked: list[PullRequest] = []
    for pr in prs:
        if pr.tickets:
            for ticket in pr.tickets:
                ticket_groups[ticket].append(pr)
        elif include_unlinked:
            unlinked.append(pr)

    digest_items: list[DigestItem] = []
    for ticket_id, grouped_prs in ticket_groups.items():
        categories = [categorize_pr(pr) for pr in grouped_prs]
        digest_items.append(
            DigestItem(
                ticket_id=ticket_id,
                ticket=None,
                prs=grouped_prs,
                category=pick_category(categories),
            ),
        )
    return digest_items, unlinked


def build_llm_payload(
    digest_items: list[DigestItem],
    unlinked: list[PullRequest],
    window_start: str,
    window_end: str,
) -> JSONDict:
    """Build the payload for narrative generation."""
    items: list[JSONDict] = []
    for item in digest_items:
        tickets = sorted({ticket for pr in item.prs for ticket in pr.tickets})
        ticket_payload = None
        if item.ticket:
            ticket_payload = {
                "identifier": item.ticket.identifier,
                "title": item.ticket.title,
                "state": item.ticket.state,
                "team": item.ticket.team,
                "project": item.ticket.project,
            }
        items.append(
            {
                "category": item.category,
                "tickets": tickets,
                "ticket_details": ticket_payload,
                "prs": [
                    {
                        "title": pr.title,
                        "summary": pr.summary or pr.title,
                        "number": pr.number,
                        "repo": pr.repo,
                        "author": pr.author,
                    }
                    for pr in item.prs
                ],
            },
        )
    return {
        "window": {
            "start": window_start,
            "end": window_end,
        },
        "items": items,
        "unlinked_prs": [
            {"title": pr.title, "repo": pr.repo, "author": pr.author}
            for pr in unlinked
        ],
    }


def call_openai(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    model: str,
    usage_tracker: UsageTracker | None,
) -> str:
    """Call OpenAI and return the response content."""
    delay = 20
    for attempt in range(1, OPENAI_MAX_ATTEMPTS + 1):
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            elapsed = f"{time.perf_counter() - start:.2f}s"
            if usage_tracker is not None and response.usage is not None:
                usage_tracker.add(response.usage)
            usage_fields = {
                "prompt_tokens": usage_int(response.usage, "prompt_tokens"),
                "completion_tokens": usage_int(response.usage, "completion_tokens"),
                "total_tokens": usage_int(response.usage, "total_tokens"),
            }
            usage_fields = {
                key: value for key, value in usage_fields.items() if value is not None
            }
            logger.info(
                "LLM call completed",
                model=model,
                elapsed=elapsed,
                **usage_fields,
            )
            content = response.choices[0].message.content
            if content is None:
                raise OpenAIEmptyResponseError
            return content.strip()
        except RateLimitError:
            if attempt == OPENAI_MAX_ATTEMPTS:
                raise
            logger.warning(
                "Rate limited by OpenAI, retrying",
                model=model,
                wait_seconds=delay,
            )
            time.sleep(delay)
    raise OpenAIRetryError


def generate_llm_narrative(
    client: OpenAI,
    digest_payload: JSONDict,
    model: str,
    usage_tracker: UsageTracker | None,
) -> str:
    """Generate the narrative using the LLM."""
    user_prompt = (
        f"{NARRATIVE_USER_PROMPT_PREFIX}{json.dumps(digest_payload, indent=2)}"
    )
    return call_openai(
        client,
        NARRATIVE_SYSTEM_PROMPT,
        user_prompt,
        model,
        usage_tracker,
    )


def format_for_slack(
    narrative: str,
    tickets_by_id: dict[str, LinearTicket],
    gh_org: str = "",
) -> str:
    """Convert LLM narrative to Slack mrkdwn format."""
    # Convert markdown headings (## text) to Slack bold (*text*)
    text = re.sub(r"^#{1,3}\s+(.+)$", r"*\1*", narrative, flags=re.MULTILINE)

    # Replace ticket IDs with Slack links to Linear
    for ticket_id, ticket in tickets_by_id.items():
        if ticket.url:
            text = text.replace(ticket_id, f"<{ticket.url}|{ticket_id}>")

    # Replace [org/]repo#123 with GitHub PR links (strip optional org prefix)
    if gh_org:
        text = re.sub(
            r"(?:[\w.-]+/)?([\w.-]+)#(\d+)",
            rf"<https://github.com/{gh_org}/\1/pull/\2|\1#\2>",
            text,
        )

    # Replace @username with bold (unlinked)
    text = re.sub(r"@(\w[\w-]*)", r"*@\1*", text)

    return text


def trim_message(message: str) -> str:
    """Trim the message to fit Slack limits."""
    if len(message) <= MAX_SLACK_CHARS:
        return message
    return message[: MAX_SLACK_CHARS - 100] + "\n\n[truncated]"


def chunk_slack_text(message: str) -> list[str]:
    """Split message into Slack block-sized chunks on newline boundaries."""
    if not message:
        return [""]
    lines = message.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        # +1 accounts for the newline joining character
        added = len(line) + (1 if current else 0)
        if current and current_len + added > SLACK_BLOCK_TEXT_LIMIT:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += added
    if current:
        chunks.append("\n".join(current))
    return chunks


def post_to_slack(webhook_url: str, message: str, window_text: str) -> None:
    """Post a message to Slack via webhook."""
    header_blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Daily Narrative Digest"},
        },
        {"type": "context", "elements": [{"type": "mrkdwn", "text": window_text}]},
        {"type": "divider"},
    ]
    max_sections = SLACK_MAX_BLOCKS - len(header_blocks)
    chunks = chunk_slack_text(message)
    if max_sections <= 0:
        sections = []
    elif len(chunks) <= max_sections:
        sections = chunks
    else:
        sections = chunks[: max_sections - 1]
        sections.append("[truncated]")
    blocks = header_blocks + [
        {"type": "section", "text": {"type": "mrkdwn", "text": section}}
        for section in sections
    ]
    response = requests.post(
        webhook_url,
        json={"text": trim_message(message), "blocks": blocks},
        timeout=30,
    )
    if response.status_code >= HTTP_ERROR_THRESHOLD:
        raise SlackWebhookError(response.status_code, response.text)


def enrich_with_linear(
    settings: Settings,
    digest_items: list[DigestItem],
) -> dict[str, LinearTicket]:
    """Resolve Linear tickets for digest items."""
    cache: dict[str, LinearTicket] = {}
    for item in digest_items:
        ticket_id = item.ticket_id
        if not ticket_id:
            continue
        if ticket_id not in cache:
            cache[ticket_id] = fetch_linear_ticket(settings, ticket_id)
        item.ticket = cache[ticket_id]
    return cache


def summarize_pr_with_llm(
    client: OpenAI,
    settings: Settings,
    pr: PullRequest,
    tickets: list[LinearTicket],
    usage_tracker: UsageTracker | None,
) -> str:
    """Summarize a PR using the LLM."""
    ticket_payload = [
        {
            "identifier": ticket.identifier,
            "title": ticket.title,
            "state": ticket.state,
            "team": ticket.team,
            "project": ticket.project,
            "found": ticket.found,
        }
        for ticket in tickets
    ]
    user_payload = {
        "pr": {
            "title": pr.title,
            "body": pr.body,
            "repo": pr.repo,
            "commits": [commit.model_dump() for commit in pr.commits],
        },
        "tickets": ticket_payload,
    }
    user_prompt = f"{PR_SUMMARY_USER_PROMPT_PREFIX}{json.dumps(user_payload, indent=2)}"
    return call_openai(
        client,
        PR_SUMMARY_SYSTEM_PROMPT,
        user_prompt,
        settings.openai_model,
        usage_tracker,
    )


def summarize_prs(
    client: OpenAI,
    settings: Settings,
    prs: list[PullRequest],
    tickets_by_id: dict[str, LinearTicket],
    usage_tracker: UsageTracker | None,
) -> None:
    """Summarize PRs in parallel and store summaries in-place."""
    if not prs:
        return
    total = len(prs)

    def task(pr: PullRequest, index: int) -> tuple[int, str]:
        """Summarize a PR and return its index and summary."""
        start = time.perf_counter()
        tickets = [
            tickets_by_id[ticket_id]
            for ticket_id in pr.tickets
            if ticket_id in tickets_by_id
        ]
        summary = summarize_pr_with_llm(
            client,
            settings,
            pr,
            tickets,
            usage_tracker,
        )
        elapsed = time.perf_counter() - start
        logger.info(
            "PR summarized",
            repo=pr.repo,
            number=pr.number,
            index=index + 1,
            total=total,
            elapsed=f"{elapsed:.2f}s",
        )
        return index, summary

    start_all = time.perf_counter()
    logger.info("Summarizing PRs in parallel", total=total)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(task, pr, idx) for idx, pr in enumerate(prs)]
        for future in concurrent.futures.as_completed(futures):
            index, summary = future.result()
            prs[index].summary = summary
    elapsed_all = f"{time.perf_counter() - start_all:.2f}s"
    logger.info("Summaries complete", total=total, elapsed=elapsed_all)


def main() -> int:
    """Run the digest CLI."""
    logger.info("Starting digest run")
    args = parse_args()
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    run_digest(args, settings, client)
    return 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Daily Narrative Digest")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print instead of posting to Slack",
    )
    return parser.parse_args()


def build_ticket_regex(settings: Settings) -> re.Pattern[str]:
    """Build the ticket regex."""
    ignore_case = settings.digest_ticket_regex_ignorecase
    flags = re.IGNORECASE if ignore_case else 0
    return re.compile(settings.digest_ticket_regex, flags)


def parse_repos(settings: Settings) -> list[str] | None:
    """Parse repository allowlist."""
    if not settings.digest_repos:
        return None
    return [repo.strip() for repo in settings.digest_repos.split(",") if repo.strip()]


def collect_prs(
    settings: Settings,
    since_iso: str,
    ticket_regex: re.Pattern[str],
) -> list[PullRequest]:
    """Collect and enrich PRs with ticket identifiers."""
    logger.info("Fetching merged PRs", org=settings.digest_gh_org, since=since_iso)
    start = time.perf_counter()
    all_prs = fetch_merged_prs(
        settings,
        settings.digest_gh_org,
        since_iso,
        parse_repos(settings),
    )
    log_elapsed("Fetched PRs", start, count=len(all_prs))
    for pr in all_prs:
        pr.tickets = extract_tickets(pr, ticket_regex)
    return all_prs


def build_stats(
    all_prs: list[PullRequest],
    digest_items: list[DigestItem],
    unlinked: list[PullRequest],
) -> dict[str, int]:
    """Build summary stats for logging."""
    return {
        "total_prs": len(all_prs),
        "included_prs": len(all_prs),
        "excluded_prs": 0,
        "tickets": len(digest_items),
        "unlinked_prs": len(unlinked),
    }


def run_digest(args: argparse.Namespace, settings: Settings, client: OpenAI) -> None:
    """Execute the digest workflow."""
    (
        window_start_iso,
        _window_end_iso,
        window_start_display,
        window_end_display,
    ) = iso_window(settings.digest_hours)
    logger.info(
        "Window: {start} → {end}",
        start=window_start_display,
        end=window_end_display,
    )
    logger.info("LLM model: {model}", model=settings.openai_model)
    input_price, output_price = resolve_pricing(settings)
    usage_tracker = UsageTracker(
        input_per_million=input_price,
        output_per_million=output_price,
    )
    if input_price is not None and output_price is not None:
        logger.info(
            "OpenAI pricing configured",
            input_per_million=input_price,
            output_per_million=output_price,
        )
    ticket_regex = build_ticket_regex(settings)
    all_prs = collect_prs(settings, window_start_iso, ticket_regex)
    window_line = f"Release window: {window_start_display} → {window_end_display}"
    if not all_prs:
        message = "No merged PRs or linked Linear tickets in this window."
        logger.info("No PRs found in window")
        logger.info("OpenAI usage: {usage}", usage=usage_tracker.to_log())
        if args.dry_run or not settings.slack_webhook_url:
            logger.info("--- DRY RUN OUTPUT ---")
            logger.opt(raw=True).info(
                "{message}\n",
                message=f"{window_line}\n\n{message}",
            )
            return
        post_to_slack(settings.slack_webhook_url, message, window_line)
        logger.info("Posted digest to Slack")
        return
    digest_items, unlinked = group_digest_items(
        prs=all_prs,
        include_unlinked=settings.digest_include_unlinked,
    )
    start = time.perf_counter()
    tickets_by_id = enrich_with_linear(settings, digest_items)
    log_elapsed("Fetched Linear tickets", start, count=len(tickets_by_id))
    start = time.perf_counter()
    summarize_prs(client, settings, all_prs, tickets_by_id, usage_tracker)
    log_elapsed("Summarized PRs", start, count=len(all_prs))
    digest_payload = build_llm_payload(
        digest_items,
        unlinked,
        window_start_display,
        window_end_display,
    )
    narrative = generate_llm_narrative(
        client,
        digest_payload,
        settings.openai_model,
        usage_tracker,
    )
    narrative = format_for_slack(narrative, tickets_by_id, settings.digest_gh_org)
    message = trim_message(narrative)
    stats = build_stats(all_prs, digest_items, unlinked)
    logger.info("Digest stats: {stats}", stats=json.dumps(stats, indent=2))
    logger.info("OpenAI usage: {usage}", usage=usage_tracker.to_log())
    if args.dry_run or not settings.slack_webhook_url:
        logger.info("--- DRY RUN OUTPUT ---")
        logger.opt(raw=True).info(
            "{message}\n",
            message=f"{window_line}\n\n{message}",
        )
        return
    post_to_slack(settings.slack_webhook_url, message, window_line)
    logger.info("Posted digest to Slack")


if __name__ == "__main__":
    sys.exit(main())
