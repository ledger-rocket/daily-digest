# Daily Digest

Daily Narrative Digest that summarizes merged PRs, enriches with Linear tickets,
and posts a narrative to Slack.

## Setup

### Required secrets (GitHub Actions)

- `DIGEST_APP_PRIVATE_KEY` - GitHub App private key (PEM) used to mint tokens.
  - Get it: Follow GitHub’s official “Managing private keys for GitHub Apps” instructions: <https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/managing-private-keys-for-github-apps>
- `LINEAR_API_KEY`
  - Get it: See Linear “Authorization” docs (personal API keys): <https://developers.linear.app/docs/graphql/authorization>
- `SLACK_WEBHOOK_URL`
  - Get it: Follow Slack Incoming Webhooks guide: <https://docs.slack.dev/messaging/sending-messages-using-incoming-webhooks/>
- `OPENAI_API_KEY`
  - Get it: See OpenAI authentication docs: <https://platform.openai.com/docs/api-reference/authentication>

### Required variables (GitHub Actions)

- `DIGEST_APP_ID` - GitHub App ID used for token generation.
  - Get it: GitHub App settings page for the Digest app (see GitHub App docs): <https://docs.github.com/en/apps/creating-github-apps/registering-a-github-app>
- `DIGEST_GH_ORG` - GitHub org to scan.
  - Value: your GitHub org slug (e.g., `ledger-rocket`).

### Optional variables (GitHub Actions)

- `DIGEST_REPOS` - comma-separated repo allowlist (empty means org-wide).
- `DIGEST_HOURS` - hours back to include (default: `24`).
- `DIGEST_INCLUDE_UNLINKED` - include unlinked PRs (default: `true`).
- `DIGEST_TICKET_REGEX` - ticket regex (default: `([A-Z][A-Z0-9]+-\d+)`).
- `DIGEST_TICKET_REGEX_IGNORECASE` - case-insensitive ticket matching (default: `true`).
- `OPENAI_MODEL` - model name for OpenAI (default: `gpt-5.2`).
- `OPENAI_PRICE_INPUT_PER_MILLION` - input token price per 1M tokens.
- `OPENAI_PRICE_OUTPUT_PER_MILLION` - output token price per 1M tokens.

## Local run

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

## More docs

See `docs/daily-digest.md` for details.
