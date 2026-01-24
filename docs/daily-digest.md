# Daily Narrative Digest

This repo includes a daily digest generator that summarizes merged PRs across GitHub, enriches them with Linear tickets, and posts a narrative to Slack. It pulls commit messages (titles + bodies) and uses the OpenAI SDK to summarize each PR before generating the overall narrative.

## Files

- `scripts/digest.py` - CLI script for manual runs and GitHub Actions.
- `.github/workflows/daily-digest.yml` - scheduled workflow.

## Setup

### Required secrets

- `LINEAR_API_KEY`
- `SLACK_WEBHOOK_URL`
- `OPENAI_API_KEY`

### Required variables

- `DIGEST_GH_ORG` - GitHub org to scan.
- `DIGEST_APP_ID` - GitHub App ID for token generation (used by the workflow).

### Optional variables

- `DIGEST_REPOS` - comma-separated repo allowlist (empty means org-wide).
- `DIGEST_HOURS` - hours back to include (default: `24`).
- `DIGEST_INCLUDE_UNLINKED` - include unlinked PRs (default: `true`).
- `DIGEST_TICKET_REGEX` - ticket regex (default: `([A-Z][A-Z0-9]+-\d+)`).
- `DIGEST_TICKET_REGEX_IGNORECASE` - case-insensitive ticket matching (default: `true`).
- `OPENAI_MODEL` - model name for OpenAI (default: `gpt-5.2`).
- `OPENAI_PRICE_INPUT_PER_MILLION` - input token price per 1M tokens (overrides Flex defaults).
- `OPENAI_PRICE_OUTPUT_PER_MILLION` - output token price per 1M tokens (overrides Flex defaults).

Flex pricing defaults (per 1M tokens):

```text
gpt-5.2: input $0.875, output $7.00
gpt-5.1: input $0.625, output $5.00
gpt-5: input $0.625, output $5.00
gpt-5-mini: input $0.125, output $1.00
gpt-5-nano: input $0.025, output $0.20
o3: input $1.00, output $4.00
o4-mini: input $0.55, output $0.138
```

## Conventions

- Include Linear IDs in PR titles, e.g. `feat(PRO-123): add audit export`. Ticket matching is case-insensitive by default.
- Ticket numbers like `PRO-000` are treated as unlinked and excluded from Linear lookups.

## Local run

Install dependencies with uv:

```bash
uv sync
```

```bash
DIGEST_GH_ORG=your-org \
LINEAR_API_KEY=... \
SLACK_WEBHOOK_URL=... \
OPENAI_API_KEY=... \
uv run python scripts/digest.py --dry-run
```


If `SLACK_WEBHOOK_URL` is omitted or `--dry-run` is used, output prints to stdout.

## GitHub Actions

The workflow runs daily and can be manually dispatched. It uses `GITHUB_TOKEN` for GitHub GraphQL, and secrets for Linear/Slack/OpenAI. The OpenAI key is required to generate PR summaries and the narrative.

Set org-wide defaults with repo variables (Settings → Secrets and variables → Actions):

- `DIGEST_GH_ORG`
- `DIGEST_REPOS`
- `DIGEST_HOURS`
- `DIGEST_INCLUDE_UNLINKED`
- `DIGEST_TICKET_REGEX`
- `OPENAI_MODEL`

## Slack formatting

The digest posts a short narrative using Slack Block Kit (header + section). URLs are omitted from the output.
