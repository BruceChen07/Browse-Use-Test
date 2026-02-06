# SNOW KB Storage Solution

This project automates the process of downloading Knowledge Base (KB) articles from ServiceNow (SNOW) as high-quality PDF documents and full-page screenshots. It leverages the [Browser Use](https://github.com/browser-use/browser-use) library and Chrome DevTools Protocol (CDP) to handle complex page rendering, lazy loading, and scrolling issues.

## Features

- **Automated Login & Navigation**: Can handle manual login flows or automated navigation to specific KB articles.
- **High-Fidelity Capture**: 
  - **Full-Page PDF**: Generates searchable, vector-based PDFs of the entire article without scrollbars.
  - **Full-Page Screenshot**: Captures the complete rendered page as a PNG image.
- **Smart Expansion**: Automatically expands scrollable containers and iframes to ensure no content is hidden.
- **Lazy Loading Handling**: Pre-scrolls and waits for dynamic content to load before capturing.

## Prerequisites

- **Windows OS** (Recommended for PowerShell script support)
- **Python 3.11+**
- **Microsoft Edge** (Installed in default location)
- **uv** (Optional, for fast dependency management) or standard `pip`.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd "SNOW KB Storage Solution"
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Or if using uv:*
    ```bash
    uv sync
    ```

3.  **Install Playwright Browsers**:
    ```bash
    playwright install
    ```

## Usage

### One-Click Execution (Recommended)

Use the provided PowerShell script to automatically start Edge in debug mode and run the downloader:

```powershell
.\run_kb_download.ps1
```

This script will:
1.  Launch Microsoft Edge with remote debugging enabled (Port 9222).
2.  Set necessary environment variables (Target URL, login mode, etc.).
3.  Execute the `KBDownload.py` Python script.

### Manual Configuration

You can modify `run_kb_download.ps1` to change the target KB article URL:

```powershell
$env:MANUAL_LOGIN_URL = "https://marsprod.service-now.com/kb_view.do?sys_kb_id=YOUR_KB_ID_HERE"
```

## Project Structure

- `KBDownload.py`: Core Python script containing the logic for browser control, page expansion, and capturing.
- `run_kb_download.ps1`: PowerShell wrapper for easy execution and environment setup.
- `requirements.txt`: Python dependency list.
- `Results/`: Directory where downloaded PDFs and Screenshots are stored (ignored by git).

## License

MIT License