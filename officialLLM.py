import asyncio
import inspect
import os
import re
import subprocess
import time
import urllib.error
import urllib.request

from browser_use import Agent, Browser, ChatBrowserUse

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


def _str_env(name: str, default: str | None = None) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    v = raw.strip()
    return v or default


def _sanitize_url(url: str | None) -> str | None:
    if not url:
        return None
    v = url.strip()
    if not v:
        return None
    v = v.strip('"\'')
    v = v.strip('`')
    return v.strip() or None


async def _wait_cdp_ready(cdp_http_url: str, timeout_sec: int) -> None:
    deadline = time.monotonic() + max(1, timeout_sec)
    url = cdp_http_url.rstrip('/') + '/json/version'
    last_err: Exception | None = None

    while time.monotonic() < deadline:
        try:
            def _probe() -> None:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=2) as resp:
                    resp.read(1)

            await asyncio.to_thread(_probe)
            return
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.2)

    if last_err is not None:
        raise last_err


def _pick_edge_executable() -> str | None:
    candidates = [
        _str_env("EDGE_EXECUTABLE_PATH"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


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
    edge_proc: subprocess.Popen[bytes] | None = None

    demo_mode = _bool_env("DEMO_MODE", False)

    cdp_url = _sanitize_url(os.getenv("BROWSER_CDP_URL") or os.getenv("CHROME_CDP_URL"))
    auto_launch_edge = _bool_env("AUTO_LAUNCH_EDGE", False) or _bool_env("BROWSER_AUTO_LAUNCH_EDGE", False) or demo_mode

    edge_port = _int_env("EDGE_REMOTE_DEBUGGING_PORT", 9222)
    edge_user_data_dir = _str_env("EDGE_USER_DATA_DIR", r"C:\temp\edge-cdp-profile")

    if auto_launch_edge and not cdp_url:
        edge_exe = _pick_edge_executable()
        if not edge_exe:
            raise RuntimeError('Edge executable not found. Set EDGE_EXECUTABLE_PATH to msedge.exe')

        args = [
            edge_exe,
            f"--remote-debugging-port={edge_port}",
            f"--user-data-dir={edge_user_data_dir}",
        ]
        edge_proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cdp_url = f"http://127.0.0.1:{edge_port}"
        await _wait_cdp_ready(cdp_url, _int_env("CDP_WAIT_TIMEOUT_SEC", 15))

    browser_kwargs: dict[str, object] = {}
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

    default_task = "Download service now KB [BAI | A2A & B2B ] BAI Operations A2A & B2B Operations from service now page https://marsprod.service-now.com/kb_view.do?sys_kb_id=31a761fc93f23e9c4ea774f86cba10ae, including the attachments"
    task = os.getenv("BROWSER_TASK") or default_task

    manual_login = _bool_env("MANUAL_LOGIN", False) or _bool_env("WAIT_FOR_LOGIN", False) or demo_mode
    manual_login_url = _sanitize_url(os.getenv("MANUAL_LOGIN_URL")) or _extract_first_url(task) or _extract_first_url(default_task)
    manual_login_navigate = _bool_env("MANUAL_LOGIN_NAVIGATE", not bool(cdp_url)) or demo_mode

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

    agent = Agent(
        task=agent_task,
        llm=llm,
        browser=browser,
        llm_timeout=_int_env("LLM_TIMEOUT_SEC", 120 * 5),
        step_timeout=_int_env("STEP_TIMEOUT_SEC", 120 * 5),
        vision_detail_level=(os.getenv("VISION_DETAIL_LEVEL") or "low").strip(),
        llm_screenshot_size=_tuple2_int_env("LLM_SCREENSHOT_SIZE", (1280, 720)),
        directly_open_url=not manual_login,
    )

    try:
        return await agent.run()
    finally:
        stop = getattr(browser, "stop", None)
        if callable(stop):
            result = stop()
            if inspect.isawaitable(result):
                await result
        else:
            close = getattr(browser, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result

        if edge_proc is not None and _bool_env("EDGE_KILL_ON_EXIT", False):
            try:
                edge_proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(example())