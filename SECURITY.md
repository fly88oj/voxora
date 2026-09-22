# Security policy

## Supported versions

Only the latest release line receives security fixes.

## Reporting a vulnerability

Please open a **private** security advisory via GitHub's *Report a vulnerability* flow on the Security tab, or contact the maintainers through the repository's issue tracker marking the issue as a security concern. Please do not open public issues with exploit details.

Expect an acknowledgement within 7 days.

## Scope notes specific to this project

- The API server is designed for **local or trusted-network research use**. It performs CPU-heavy inference on untrusted input sizes; if you expose it publicly, put it behind a rate-limiting proxy and set `VOXORA_API_KEY`.
- Audio is decoded with libsndfile via `soundfile`; keep that system library updated for decoder CVEs.
- The optional API key is read from the `VOXORA_API_KEY` environment variable only. No credentials are ever written to source, examples, or tests.
