import asyncio
import inspect
import os
import re
from datetime import datetime
from pathlib import Path

from browser_use import Agent, Browser, ChatBrowserUse, Controller
from browser_use.browser.session import BrowserSession
from cdp_use.cdp.page import CaptureScreenshotParameters, PrintToPDFParameters, Viewport
from pydantic import BaseModel
import base64

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except Exception:
        return default


def _tuple2_int_env(name: str, default: tuple[int, int]) -> tuple[int, int]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    sep = "x" if "x" in raw.lower() else ","
    parts = [p.strip() for p in raw.lower().split(sep) if p.strip()]
    if len(parts) != 2:
        return default
    try:
        return int(parts[0]), int(parts[1])
    except Exception:
        return default


def _extract_first_url(text: str) -> str | None:
    m = re.search(r"https?://[^\s)\]>\"']+", text)
    if not m:
        return None
    return m.group(0).rstrip(".,;)")


async def _await_enter(prompt: str) -> None:
    await asyncio.to_thread(input, prompt)


async def example():
    project_root = Path(__file__).resolve().parent
    results_root = Path(os.getenv("RESULTS_DIR") or (project_root / "Results"))
    run_id = (os.getenv("RESULTS_RUN_ID") or "").strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / run_id
    downloads_dir = run_dir / "downloads"
    extracted_dir = run_dir / "extracted"

    downloads_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    cdp_url = (os.getenv("BROWSER_CDP_URL") or os.getenv("CHROME_CDP_URL") or "").strip() or None

    browser_kwargs: dict[str, object] = {"downloads_path": str(downloads_dir)}
    if cdp_url:
        browser_kwargs["cdp_url"] = cdp_url
        browser_kwargs["is_local"] = True
        browser_kwargs["keep_alive"] = True
    else:
        browser_kwargs["headless"] = _bool_env("BROWSER_HEADLESS", False)

        browser_channel = (os.getenv("BROWSER_CHANNEL") or "").strip() or None
        if browser_channel:
            browser_kwargs["channel"] = browser_channel

        executable_path = (os.getenv("BROWSER_EXECUTABLE_PATH") or "").strip() or None
        if executable_path:
            browser_kwargs["executable_path"] = executable_path

        user_data_dir = (os.getenv("BROWSER_USER_DATA_DIR") or "").strip() or None
        if user_data_dir:
            browser_kwargs["user_data_dir"] = user_data_dir

        if _bool_env("BROWSER_KEEP_ALIVE", False):
            browser_kwargs["keep_alive"] = True

    browser = Browser(**browser_kwargs)
    llm = ChatBrowserUse()

    task = (
        os.getenv("BROWSER_TASK")
        or "Download service now KB [BAI | A2A & B2B ] BAI Operations A2A & B2B Operations from service now page https://marsprod.service-now.com/kb_view.do?sys_kb_id=31a761fc93f23e9c4ea774f86cba10ae, including the attachments"
    )

    manual_login = _bool_env("MANUAL_LOGIN", False) or _bool_env("WAIT_FOR_LOGIN", False)
    manual_login_url = (os.getenv("MANUAL_LOGIN_URL") or "").strip() or _extract_first_url(task)
    manual_login_navigate = _bool_env("MANUAL_LOGIN_NAVIGATE", not bool(cdp_url))

    if manual_login:
        start = getattr(browser, "start", None)
        if callable(start):
            try:
                await start()
            except Exception:
                if cdp_url:
                    print(
                        "CDP connection failed. Ensure your Edge/Chrome is started with --remote-debugging-port and that this URL is reachable: "
                        + cdp_url.rstrip("/")
                        + "/json/version"
                    )
                raise

        if manual_login_navigate and manual_login_url:
            navigate_to = getattr(browser, "navigate_to", None)
            if callable(navigate_to):
                await navigate_to(manual_login_url, new_tab=False)

        get_url = getattr(browser, "get_current_page_url", None)
        if callable(get_url):
            try:
                current_url = await get_url()
                print(f"Current page before login: {current_url}")
            except Exception:
                pass

        prompt = (
            os.getenv("MANUAL_LOGIN_PROMPT")
            or "Please complete SNOW login in the opened browser (including any MFA/SSO steps).\nWhen you are fully logged in and ready to continue, press Enter here to resume... "
        )
        await _await_enter(prompt)

        get_url = getattr(browser, "get_current_page_url", None)
        if callable(get_url):
            try:
                current_url = await get_url()
                print(f"Current page after login: {current_url}")
            except Exception:
                pass

    agent_task = (
        os.getenv("POST_LOGIN_TASK")
        or (
            "You are already logged in. Continue from the current browser tab/session without performing any login steps. "
            + task
        )
        if manual_login
        else task
    )

    def _float_env(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return float(raw.strip())
        except Exception:
            return default

    download_instruction = (
        "\nIMPORTANT INSTRUCTIONS:\n"
        "1. SAVE KB CONTENT: You MUST save the full content of the KB article.\n"
        "   - PRIMARY: Use 'Save as PDF' tool to save as 'kb_article.pdf'.\n"
        "   - SECONDARY: Use 'Save a full page screenshot' tool to save as 'kb_article.png'.\n"
        "   - Execute BOTH tools to ensure we have a backup format.\n"
        "   - Do NOT extract text to markdown files.\n"
        "2. DOWNLOAD ATTACHMENTS: After saving content, look for attachments.\n"
        "3. You MUST click download links ONE BY ONE.\n"
        "4. After clicking a download link, you MUST WAIT for the download to complete before clicking the next one.\n"
        "5. Do NOT click multiple download links in the same step.\n"
        "6. Verify that the file has been downloaded before proceeding.\n"
        "7. If there are multiple attachments, repeat the process for each one individually.\n"
    )

    full_task = agent_task + download_instruction

    # Define custom tool for PDF
    class SavePdfArgs(BaseModel):
        filename: str = "kb_article.pdf"

    controller = Controller()

    async def _expand_page_content(cdp_session):
        """Helper to force expand all scrollable elements on the page."""
        js_script = """
        (async function() {
            const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
            
            // 1. Scroll to bottom to load lazy images/content
            try {
                window.scrollTo(0, document.body.scrollHeight);
                await sleep(1000); 
            } catch(e) {}

            // 2. Define expansion logic
            function expand(doc) {
                try {
                    // Expand main doc
                    doc.body.style.height = 'auto';
                    doc.body.style.minHeight = doc.body.scrollHeight + 'px';
                    doc.body.style.overflow = 'visible'; 
                    doc.documentElement.style.height = 'auto';
                    doc.documentElement.style.minHeight = doc.documentElement.scrollHeight + 'px';
                    doc.documentElement.style.overflow = 'visible';

                    // Expand all scrollable elements
                    const all = doc.querySelectorAll('*');
                    for (const el of all) {
                        try {
                            // Check if element is an iframe
                            if (el.tagName === 'IFRAME') {
                                try {
                                    if (el.contentDocument) {
                                        expand(el.contentDocument);
                                        // Add buffer to height
                                        el.style.height = (el.contentDocument.documentElement.scrollHeight + 50) + 'px';
                                        continue;
                                    }
                                } catch(e) {}
                            }

                            // Check for scrollbars or overflow
                            const style = window.getComputedStyle(el);
                            if (el.scrollHeight > el.clientHeight || 
                                ['auto', 'scroll'].includes(style.overflowY)) {
                                
                                el.style.height = 'auto';
                                el.style.minHeight = el.scrollHeight + 'px';
                                el.style.maxHeight = 'none';
                                el.style.overflow = 'visible';
                                el.style.overflowY = 'visible';
                            }
                        } catch(e) {}
                    }
                } catch(e) { console.error(e); }
            }

            // 3. Run expansion
            expand(document);
            await sleep(1500); // Wait for layout to settle

            // 4. Scroll back to top
            window.scrollTo(0, 0);
            await sleep(500); // Wait for scroll
        })();
        """
        try:
            # Use awaitPromise=True to ensure JS finishes (including sleeps) before Python continues
            await cdp_session.cdp_client.send_raw(
                "Runtime.evaluate", 
                params={"expression": js_script, "awaitPromise": True}, 
                session_id=cdp_session.session_id
            )
            # Extra buffer in Python just in case
            await asyncio.sleep(1) 
        except Exception:
            pass

    @controller.registry.action("Save as PDF", param_model=SavePdfArgs)
    async def save_pdf(params: SavePdfArgs, browser_session: BrowserSession):
        focused_target = browser_session.get_focused_target()
        if not focused_target:
            page_targets = browser_session.get_page_targets()
            if not page_targets:
                return "No page found"
            target_id = page_targets[-1].target_id
        else:
            target_id = focused_target.target_id
        
        cdp_session = await browser_session.get_or_create_cdp_session(target_id)
        
        # Expand content before printing
        await _expand_page_content(cdp_session)

        # Print to PDF
        pdf_params = PrintToPDFParameters(
            printBackground=True,
            marginTop=0,
            marginBottom=0,
            marginLeft=0,
            marginRight=0,
            paperWidth=8.27, # A4
            paperHeight=11.69 # A4
        )
        
        try:
            # Try typed call first, fallback to raw if needed
            if hasattr(cdp_session.cdp_client.send.Page, "printToPDF"):
                result = await cdp_session.cdp_client.send.Page.printToPDF(params=pdf_params, session_id=cdp_session.session_id)
            else:
                result = await cdp_session.cdp_client.send_raw("Page.printToPDF", params=pdf_params, session_id=cdp_session.session_id)
            
            if result and 'data' in result:
                filepath = extracted_dir / params.filename
                with open(filepath, "wb") as f:
                    f.write(base64.b64decode(result['data']))
                return f"PDF saved to {filepath}"
            return "Failed to save PDF (no data)"
        except Exception as e:
            return f"Failed to print PDF: {e}"

    # Define custom tool for screenshot
    class SaveScreenshotArgs(BaseModel):
        filename: str = "kb_article.png"

    @controller.registry.action("Save a full page screenshot", param_model=SaveScreenshotArgs)
    async def save_screenshot(params: SaveScreenshotArgs, browser_session: BrowserSession):
        focused_target = browser_session.get_focused_target()
        if not focused_target:
            page_targets = browser_session.get_page_targets()
            if not page_targets:
                return "No page found"
            target_id = page_targets[-1].target_id
        else:
            target_id = focused_target.target_id

        cdp_session = await browser_session.get_or_create_cdp_session(target_id)
        
        # Expand content before screenshot
        await _expand_page_content(cdp_session)

        try:
            # Get Layout Metrics
            metrics = await cdp_session.cdp_client.send_raw("Page.getLayoutMetrics", session_id=cdp_session.session_id)
            
            # Check for contentSize (standard) or cssContentSize (some versions)
            content_size = metrics.get('contentSize') or metrics.get('cssContentSize')
            if not content_size:
                return "Failed to get layout metrics for full page. Try PDF instead."
            
            width = content_size['width']
            height = content_size['height']
            
            clip = Viewport(x=0, y=0, width=width, height=height, scale=1)
            
            screenshot_params = CaptureScreenshotParameters(
                format='png',
                captureBeyondViewport=True,
                clip=clip,
                fromSurface=True
            )
            
            result = await cdp_session.cdp_client.send.Page.captureScreenshot(params=screenshot_params, session_id=cdp_session.session_id)
            
            if result and 'data' in result:
                filepath = extracted_dir / params.filename
                with open(filepath, "wb") as f:
                    f.write(base64.b64decode(result['data']))
                return f"Screenshot saved to {filepath}"
            return "Failed to save screenshot"
            
        except Exception as e:
            return f"Error taking screenshot: {e}"

    agent = Agent(
        task=full_task, 
        llm=llm,
        browser=browser,
        controller=controller,
        llm_timeout=_int_env("LLM_TIMEOUT_SEC", 120 * 5),
        step_timeout=_int_env("STEP_TIMEOUT_SEC", 120 * 5),
        vision_detail_level=(os.getenv("VISION_DETAIL_LEVEL") or "low").strip(),
        llm_screenshot_size=_tuple2_int_env("LLM_SCREENSHOT_SIZE", (1912, 948)),
        directly_open_url=not manual_login,
        file_system_path=str(extracted_dir),
        max_actions_per_step=_int_env("MAX_ACTIONS_PER_STEP", 1),
    )

    try:
        await agent.run()
    finally:
        stop = getattr(browser, "stop", None)
        if callable(stop):
            result = stop()
            if inspect.isawaitable(result):
                await result
            # Wait for underlying Windows pipes to close properly
            await asyncio.sleep(1)
            return

        close = getattr(browser, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(example())
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        # Ignore specific Windows asyncio error about pending operations at deallocation
        if "pending operation at deallocation" not in str(e) and "Event loop is closed" not in str(e):
            raise
    finally:
        try:
            # Cancel all running tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            
            # Allow cancelled tasks to cleanup
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                
            loop.close()
        except Exception:
            # Suppress any errors during loop closure (common on Windows)
            pass