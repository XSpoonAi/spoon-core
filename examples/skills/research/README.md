# Research skill

Instructions for the agent are in `SKILL.md`. The skill registers three web
search scripts in `scripts/`, which the skill executor runs with only the
environment variables in `ScriptExecutor.env_passthrough`.

## tavily_search

Requires `TAVILY_API_KEY` and the `tavily-python` package.

## keenable_search (opt-in)

Disabled by default. Until you set `KEENABLE_SEARCH_ENABLED=1`, the script
returns an error and sends nothing over the network.

```bash
export KEENABLE_SEARCH_ENABLED=1
export KEENABLE_API_KEY=your-keenable-api-key   # optional
```

- Privacy: when enabled, each search query is sent to Keenable's API at
  `api.keenable.ai`, with an `X-Keenable-Title: spoon-core` header and your API
  key if you set one. Keenable's [privacy policy](https://keenable.ai/privacy)
  says it may create de-identified data from what it collects and use it to
  improve its services. See also the [terms](https://keenable.ai/terms).
- Rate limits: without a key, 1,000 requests per hour and 10 per second per
  IP address. Requests over the limit get HTTP 429, which the script returns
  as an error. An API key lifts these limits.
- API key: without `KEENABLE_API_KEY` the script calls
  `POST https://api.keenable.ai/v1/search/public` with no key. With it, the
  script calls `POST https://api.keenable.ai/v1/search` and sends the key in
  the `X-API-Key` header. Both variables are in the executor's default
  `env_passthrough`. API reference: https://docs.keenable.ai/api-reference

## youcom_search (opt-in)

Disabled by default. Until you set `YOUCOM_SEARCH_ENABLED=1`, the script
returns an error and sends nothing over the network.

```bash
export YOUCOM_SEARCH_ENABLED=1
export YDC_API_KEY=your-you-com-api-key   # optional
```

- Privacy: when enabled, each search query is sent to You.com
  (`api.you.com`, or `ydc-index.io` if you set a key), with a
  `User-Agent: youdotcom-integration/XSpoonAi-spoon-core` header and your API
  key if you set one. See You.com's [privacy policy](https://you.com/privacy).
- Rate limits: without a key, the keyless endpoint allows about 100 searches
  per day and up to 50 results per query. Requests over the limit return an
  HTTP error, which the script returns as an error. An API key lifts the
  daily limit.
- API key: without `YDC_API_KEY` the script calls
  `GET https://api.you.com/v1/agents/search` with no key. With it, the script
  calls `POST https://ydc-index.io/v1/search` and sends the key in the
  `X-API-Key` header. Both variables are in the executor's default
  `env_passthrough`. API spec: https://you.com/specs/openapi_search_v1.yaml
