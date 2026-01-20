# Browser Use Agent Example

This project demonstrates how to use the [Browser Use](https://github.com/browser-use/browser-use) library to create an AI agent that controls a web browser to perform tasks.

## Features

- **Automated Browser Control**: Uses Playwright (via browser-use) to navigate and interact with websites.
- **LLM Integration**: Uses `ChatBrowserUse` (or other configured LLMs) to understand tasks and make decisions.
- **Environment Management**: Uses `uv` for fast Python package management.

## Prerequisites

- **Python**: Version 3.11 or higher (3.13 is configured in this project).
- **uv**: A fast Python package installer and resolver. [Install uv](https://docs.astral.sh/uv/getting-started/installation/).
- **Browser Use API Key**: Required for using the `ChatBrowserUse` model. Get it from [Browser Use Cloud](https://browser-use.com/).

## Installation

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <your-repo-url>
    cd BrowseUseTest
    ```

2.  **Install dependencies**:
    This project uses `uv` for dependency management.
    ```bash
    uv sync
    ```
    This will create a virtual environment (`.venv`) and install all required packages specified in `pyproject.toml`.

3.  **Install Playwright browsers**:
    The browser-use library requires Playwright browsers to be installed.
    ```bash
    uv run playwright install
    ```

## Configuration

1.  **Create a `.env` file**:
    Copy the example configuration or create a new `.env` file in the project root.
    ```bash
    # Windows (PowerShell)
    New-Item -Path .env -ItemType File
    ```

2.  **Add your API Key**:
    Open the `.env` file and add your Browser Use API key:
    ```env
    BROWSER_USE_API_KEY=your_actual_api_key_here
    ```
    *(Note: The `.env` file is git-ignored to protect your secrets.)*

## Usage

To run the agent, execute the `main.py` script using `uv`:

```bash
uv run main.py
```

### Modifying the Task
Open `main.py` and modify the `task` parameter in the `Agent` initialization to change what the agent does:

```python
agent = Agent(
    task="Your new task description here",
    llm=llm,
    browser=browser,
)
```

## Project Structure

- `main.py`: The entry point script containing the agent logic.
- `.env`: Configuration file for environment variables (API keys).
- `pyproject.toml`: Project metadata and dependencies.
- `uv.lock`: Lock file ensuring reproducible builds.

## Troubleshooting

- **Browser not opening?** Ensure you've run `uv run playwright install`.
- **API Key errors?** Double-check your `.env` file and ensure the variable name is exactly `BROWSER_USE_API_KEY`.