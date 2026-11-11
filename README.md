# Local AI Study Chat

A small browser chat app backed by Ollama. Models run locally; optional web retrieval fetches short passages from public data APIs and shows their source links. This does not train a model or copy the internet into the project.

## Features

- Select any model already installed in Ollama.
- Choose General, Navier-Stokes, or P / NP topic guidance.
- Opt in to retrieval from Wikipedia and arXiv for each question.
- Optionally add Google's official Programmable Search JSON API.
- Keep model weights on the machine running Ollama.

## Run locally

Requirements: Python 3 and [Ollama](https://ollama.com/download).

In one terminal, start Ollama if it is not already running:

```bash
ollama serve
```

In another terminal, download a model and start the app:

```bash
ollama pull qwen2.5:3b
python3 app.py
```

Open http://localhost:8001. Other installed models appear in the model selector. Model speed and answer quality depend on the model and available hardware.

## Web retrieval and privacy

When **Search sources** is enabled, the backend queries Wikipedia and arXiv's public APIs, sends a few short excerpts to the selected local model, and displays links with the answer. Queries therefore go to those providers. With search disabled, prompts are sent only to the local Ollama service by this app.

Google search is optional and requires a Google Programmable Search Engine ID and API key. Configure them as environment variables; do not put keys in source files or commit them:

```bash
export GOOGLE_CSE_ID='your-search-engine-id'
export GOOGLE_CSE_API_KEY='your-api-key'
python3 app.py
```

Google API quotas and terms apply. Mozilla is a browser organization, not a general web search API; Yahoo does not provide a dependable free public general-search API for this app. The project does not scrape either service.

The app listens on `127.0.0.1` by default. Keep it local if model files or prompts must remain private. Public hosting requires authentication, rate limiting, and a deliberate data-retention policy.

## Configuration

Set `OLLAMA_MODEL` to choose the default installed model, `OLLAMA_URL` to change the local Ollama endpoint, or `APP_PORT` to choose another app port.

## License

This project is licensed under the MIT License.
