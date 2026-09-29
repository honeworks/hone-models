# 0017: Model endpoints from the environment and a `.env` file

## Status

`accepted (2026-09-29, owner)`: "I can add Astra and Fable APIs through an OpenAI compatible source.
These should be configurable through env or things like that. Maybe even a .env file."

## Context

Hosted judges (Astra, Fable) are reached through an OpenAI-compatible gateway. Its URL and key differ
per machine and must not be committed. A registry entry can already name the variable that holds its
key (`api_key_env`), but its `base_url` has to be written into a TOML file, and every variable has to be
exported in the shell that runs the program.

## Problem

- The gateway URL cannot come from the environment, so a shared `hone-models.toml` would carry a
  machine-specific (and sometimes credential-bearing) URL.
- There is no simple place to keep the variables for a project; each shell and each tool has to set them.
- A URL with `user:password@` in it would be recorded as it is when it appears in an error message.

## Options

1. `base_url_env` on an entry plus a `.env` file read at registry load (chosen).
2. Environment interpolation inside TOML values (`base_url = "${JUDGES_BASE_URL}"`): a small template
   language with its own quoting and error rules, for one use.
3. python-dotenv: a new core dependency for about thirty lines of parsing.

## Decision

- **`base_url_env`** on a registry entry names the variable that holds the base URL, as `api_key_env`
  names the key's. When the variable is set (non-empty) its value is the entry's `base_url`; otherwise
  the entry's `base_url` is used. With the variable unset and no `base_url`, calling the model raises
  `ConfigError` naming the variable, before any request. The URL is resolved when the entry is built
  (at registry load), so everything that reads `base_url` (the `local` flag, providers) sees it.
- **`.env` files.** `mk.registry.load()` first reads the file named by `HONE_ENV_FILE` (if set; it must
  exist) and then `./.env` in the current directory (if present), each file once per process. Lines are
  `KEY=VALUE`, with an optional `export ` prefix; blank lines and lines starting with `#` are skipped; a
  value may be wrapped in single or double quotes (taken literally, no escapes); an unquoted value ends
  at ` #` (a comment). A variable already in the environment is never overwritten, so the process
  environment wins, then `HONE_ENV_FILE`, then `./.env`. A malformed line is a `ConfigError` naming the
  file and line number, never the line's text. Values are never logged.
- **Credentials in URLs.** The record sinks replace the `user:password` part of any URL with `***`, in
  addition to the existing secret rules.
- **No `model_env`.** A model name is neither a secret nor machine-specific in practice; the project
  file `./hone-models.toml` already overrides `model` per project. Left out until there is a need.

## Consequences

- A committed registry entry can point at a gateway without naming it; `.env` (git-ignored) holds the
  URL and key.
- Loading the registry can change `os.environ` (only variables that were unset), once per file per
  process. A `.env` edited while the process runs is not reread.
- Keys set through `.env` are scrubbed from records as before (`api_key_env` values are registered as
  secrets when a call reads them).

## Migration and compatibility

Additive: a registry without `base_url_env` and a directory without `.env` behave as before. A
directory that already has a `.env` for another tool now has its unset variables exported into the
process when the registry loads; a line hone-models cannot parse raises `ConfigError` (point
`HONE_ENV_FILE` elsewhere or fix the line).
