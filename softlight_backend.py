import os
import re 
import time
import json
import hashlib
from typing import TypedDict, List, Dict, Any, Optional
from urllib.parse import urlparse
import base64
from pydantic_settings import BaseSettings
import regex 
from pathlib import Path

# import re



from pydantic import BaseModel, Field, field_validator, model_validator
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError, Page, Locator
# from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class Settings(BaseSettings):
    OPENAI_API_KEY: str

    class Config:
        env_file = ".env"    
        env_file_encoding = "utf-8"

settings = Settings()



# import os
# settings = Settings()
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY




# Auth 
AUTH_DIR = os.path.abspath(".auth")
os.makedirs(AUTH_DIR, exist_ok=True)

def _host_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""
    
def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-") or "untitled"
def infer_app_name(task: str) -> str:
    t = task.lower()
    if "notion" in t:
        return "notion"
    if "linear" in t:
        return "linear"
    if "asana" in t:
        return "asana"
    # add more apps here if needed
    return "unknown-app"



def _auth_flag_path(host: str) -> str:
    safe = host.replace(":", "_")
    return os.path.join(AUTH_DIR, f"{safe}.ok")

def _has_manual_login(host: str) -> bool:
    return bool(host) and os.path.exists(_auth_flag_path(host))

def _mark_manual_login_done(host: str) -> None:
    if host:
        with open(_auth_flag_path(host), "w") as f:
            f.write(str(int(time.time())) + "\n")


# Flexible Action Schema (planner to  executor)
class Target(BaseModel):
    role: Optional[str] = None
    name: Optional[str] = None
    text: Optional[str] = None
    css: Optional[str] = None
    xpath: Optional[str] = None
    match: str = Field(default="auto")  # "exact" | "contains" | "regex" | "auto"

    @field_validator("match")
    @classmethod
    def _match_ok(cls, v: str) -> str:
        if v not in {"exact", "contains", "regex", "auto"}:
            raise ValueError("match must be one of {'exact','contains','regex','auto'}")
        return v


class PlanStep(BaseModel):
    action: str                      # "goto"|"click"|"fill"|"press"|"wait_for"|"hover"|"upload"|"finish"
    target: Optional[Target] = None
    value: Optional[str] = None
    url: Optional[str] = None
    save_state: bool = False
    description: str = ""
    done: bool = False

    @field_validator("action")
    @classmethod
    def _action_ok(cls, v: str) -> str:
        allowed = {"goto", "click", "fill", "press", "wait_for", "hover", "upload", "finish"}
        v = (v or "").strip().lower()
        if v not in allowed:
            raise ValueError(f"Unsupported action: {v}")
        return v

    @model_validator(mode="after")
    def _url_vs_target(self):
        # cross-field validation runs AFTER fields are parsed
        if self.action == "goto" and not self.url:
            raise ValueError("goto requires a non-empty 'url'")
        return self



# Strip JS-style // comments (planner sometimes adds them)
_COMMENT_LINE = re.compile(r"^\s*//.*$")
def strip_js_comments(s: str) -> str:
    return "\n".join(line for line in s.splitlines() if not _COMMENT_LINE.match(line.strip()))

def parse_plan_step(raw: str) -> PlanStep:
    cleaned = strip_js_comments(raw).strip()
    obj = json.loads(cleaned)
    if isinstance(obj, dict) and "action" in obj and "target" not in obj:
        sel = obj.get("selector")
        if isinstance(sel, str) and sel.strip():
            obj["target"] = selector_string_to_target(sel.strip())
        if obj.get("action") in {"fill", "press"} and "value" not in obj and "text" in obj:
            obj["value"] = obj["text"]
    return PlanStep.model_validate(obj)  # v2 spelling (equivalent to parse_obj)


def plan_or_retry(llm, prompt, tries: int = 2) -> PlanStep:
    """
    `prompt` can be a string OR a list of LangChain messages.
    """
    last_err = None
    for _ in range(tries + 1):
        try:
            raw = llm.invoke(prompt).content  # LangChain accepts str or message list
            return parse_plan_step(raw)
        except Exception as e:
            last_err = e
    raise last_err

def _clear_editable(self, loc):
    """Robust clear for contenteditable/inputs across apps."""
    try:
        loc.click(timeout=8000)
        # Native clear is a no-op for many CEs but cheap to try
        try:
            loc.fill("")
        except Exception:
            pass

        # Dismiss any popovers/slash menus
        try:
            self.page.keyboard.press("Escape")
            self.page.wait_for_timeout(100)
        except Exception:
            pass

        # Proper select-all chords 
        for chord in ("Meta+A", "Control+A"):
            try:
                self.page.keyboard.press(chord)
                break
            except Exception:
                continue

        self.page.keyboard.press("Backspace")
    except Exception:
        pass



# Selector helpers (string → Target, Target → Playwright Locator)
def selector_string_to_target(selector: str) -> Target:
    """
    Accepts simple strings (e.g., 'text=New page', 'role=button[name=Create]',
    'css=.btn.primary', 'xpath=//button[contains(., "Create")]') and converts to Target.
    """
    sel = selector.strip()
    # role=button[name=Something]
    if sel.startswith("role="):
        # Very small parser
        # role=button[name=Create Project]
        role = None
        name = None
        _m = re.match(r"role=([a-zA-Z]+)(?:\[name=(.+)\])?$", sel)
        if _m:
            role = _m.group(1)
            name = (_m.group(2) or "").strip()
        return Target(role=role, name=name or None)
    if sel.startswith("text="):
        return Target(text=sel[len("text="):].strip())
    if sel.startswith("css="):
        return Target(css=sel[len("css="):].strip())
    if sel.startswith("xpath="):
        return Target(xpath=sel[len("xpath="):].strip())
    # fallback: treat as CSS first, then as text if CSS fails
    if any(ch in sel for ch in ".#>:[]"):
        return Target(css=sel)
    return Target(text=sel)


def _query_textbox_like(page: Page, name: Optional[str]) -> Optional[Locator]:
    tries = []
    if name:
        tries.append(lambda: page.get_by_role("textbox", name=name))
        tries.append(lambda: page.get_by_role("textbox", name=re.compile(re.escape(name), re.I)))
        tries.append(lambda: page.locator(f'input[placeholder="{name}"]'))
        tries.append(lambda: page.locator(f'input[placeholder*="{name}"]'))
        tries.append(lambda: page.locator(f'textarea[placeholder="{name}"]'))
        tries.append(lambda: page.locator(f'textarea[placeholder*="{name}"]'))
    tries.append(lambda: page.locator('[contenteditable="true"]'))
    for attempt in tries:
        try:
            loc = attempt()
            loc.first.wait_for(timeout=1500)
            return loc.first
        except Exception:
            continue
    return None


def _sanitize_label(label: str, max_len: int = 60) -> str:
    s = (label or "").strip()
    # collapse whitespace and replace unsafe filesystem chars
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\-.]+", "_", s)   
    return s[:max_len] or "capture"

def _unique_path(dirpath: str, basename: str, ext: str) -> str:
    os.makedirs(dirpath, exist_ok=True)
    candidate = os.path.join(dirpath, f"{basename}{ext}")
    if not os.path.exists(candidate):
        return candidate
    i = 1
    while True:
        p = os.path.join(dirpath, f"{basename}_{i}{ext}")
        if not os.path.exists(p):
            return p
        i += 1


def _collect_affordances(page, limit=40):
    afford = []
    selectors = [
        "button,[role=button]",
        "a[href]",
        "[role=menuitem],[role=option]",
        "[data-testid]"
    ]
    try:
        for sel in selectors:
            loc = page.locator(sel)
            n = min(loc.count(), 80)
            for i in range(n):
                el = loc.nth(i)
                try:
                    if not el.is_visible():
                        continue
                    txt = (el.inner_text() or "").strip()
                    if 1 <= len(txt) <= 50:
                        afford.append(txt)
                except Exception:
                    continue
        seen, out = set(), []
        for t in afford:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out[:limit]
    except Exception:
        return []
    



def resolve_target(page: Page, target: Optional["Target"]) -> Optional[Locator]:
    if target is None:
        return None

    if (target.role or "").lower() == "textbox":
        loc = _query_textbox_like(page, target.name or target.text)
        if loc:
            return loc
    attempts = []
    if target.role and (target.name or target.text):
        name = target.name or target.text
        attempts.append(lambda: page.get_by_role(target.role, name=name))
        attempts.append(lambda: page.get_by_role(target.role, name=re.compile(re.escape(name), re.I)))
    elif target.role:
        attempts.append(lambda: page.get_by_role(target.role))
    if target.name:
        attempts.append(lambda: page.get_by_text(target.name))
    if target.text:
        attempts.append(lambda: page.get_by_text(target.text))
    if target.css:
        attempts.append(lambda: page.locator(target.css))
    if target.xpath:
        attempts.append(lambda: page.locator(f"xpath={target.xpath}"))
    for attempt in attempts:
        try:
            loc = attempt()
            loc.first.wait_for(timeout=1500)
            return loc.first
        except Exception:
            continue
    return None



# BrowserEnv (persistent context + robust actions + screenshots)

# class BrowserEnv:
#     def __init__(self, headless: bool = True, dataset_root: str = "datasets"):
#         self.headless = headless
#         self.dataset_root = Path(dataset_root)
#         self.playwright = None
#         self.browser = None
#         self.page = None

#     def __enter__(self):
#         self.playwright = sync_playwright().start()
#         user_data_dir = os.path.abspath("playwright_data")
#         self.browser = self.playwright.chromium.launch_persistent_context(
#             user_data_dir=user_data_dir,
#             headless=self.headless,
#         )
#         pages = self.browser.pages
#         self.page = pages[0] if pages else self.browser.new_page()
#         self.page.set_viewport_size({"width": 1440, "height": 900})
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         if self.browser:
#             self.browser.close()
#         if self.playwright:
#             self.playwright.stop()

#     def press_keys(self, sequence: str):
#         """Press keys or type text. Supports comma-separated sequences"""
#         if not sequence:
#             sequence = "Enter"
#         parts = [p.strip() for p in sequence.split(",") if p.strip()]
#         for p in parts:
#             try:
#                 # If it looks like a key or chord, press; otherwise type text
#                 if len(p) == 1 or "+" in p or p.lower() in {
#                     "enter","escape","esc","tab","shift","control","ctrl","alt","meta","command","cmd",
#                     "arrowup","arrowdown","arrowleft","arrowright","home","end","pageup","pagedown",
#                     "backspace","delete","space"
#                 }:
#                     self.page.keyboard.press(p)
#                 else:
#                     self.page.keyboard.type(p)
#             except Exception:
#                 self.page.keyboard.type(p)
#         return self._capture(f"press:{sequence}")


#     def _capture(self, label: str) -> Dict[str, Any]:
#     # Build filename first
#         ts = int(time.time() * 1000)
#         safe = _sanitize_label(label, max_len=60)
#         filename = _unique_path(self.dataset_root, f"{ts}_{safe}", ".png")

#         # Take screenshot (best-effort)
#         try:
#             os.makedirs(self.dataset_root, exist_ok=True)
#             if self.page:
#                 self.page.screenshot(path=filename, full_page=True)
#         except Exception as e:
#             print(f"[ERROR] screenshot failed: {e}")
#             filename = ""  # keep structure consistent even if screenshot fails

#         # Grab text + affordances (best-effort)
#         try:
#             text = self.page.inner_text("body")
#         except Exception:
#             text = ""

#         affordances = []
#         try:
#             affordances = _collect_affordances(self.page, limit=40)
#         except Exception:
#             pass

#         # State hash
#         url = self.page.url if self.page else ""
#         state_hash = hashlib.sha256((url + text[:2000]).encode("utf-8")).hexdigest() if url else ""

#         return {
#             "url": url,
#             "label": label,
#             "screenshot_path": filename,
#             "text_snippet": text[:800],
#             "affordances": affordances,
#             "state_hash": state_hash,
#         }

#     def wait_for_flexible(self, selector: Optional[str] = None, timeout: int = 15000):
#         """
#         Wait for a selector or general page readiness after an action.
#         - If selector is given: waits for it explicitly.
#         - If no selector: waits for network idle + main contenteditable (for editors like Notion).
#         """
#         try:
#             if selector:
#                 self.page.wait_for_selector(selector, timeout=timeout, state="visible")
#             else:
#                 # Fallback for rich editors (Notion, Linear, etc.)
#                 self.page.wait_for_load_state("networkidle", timeout=timeout)
#                 try:
#                     self.page.locator('[contenteditable="true"]').first.wait_for(timeout=timeout)
#                 except Exception:
#                     # fallback: wait for <body> to be visible
#                     self.page.wait_for_selector("body", timeout=timeout, state="visible")
#             return True
#         except Exception as e:
#             print(f"[WARN] wait_for_flexible failed: {e}")
#             return False


#     def goto(self, url: str, wait_until: str = "domcontentloaded", timeout: int = 45000):
#         try:
#             self.page.goto(url, wait_until=wait_until, timeout=timeout)
#         except PlaywrightTimeoutError as e:
#             print(f"[WARN] goto timeout for {url}: {e}")
#             # still try to settle:
#             try:
#                 self.page.wait_for_load_state("domcontentloaded", timeout=5000)
#             except Exception:
#                 pass
#         except Exception as e:
#             print(f"[ERROR] goto failed for {url}: {e}")
#         return self._capture(f"goto:{url}")
    
#     def click_flexible(self, target: "Target"):
#         loc = resolve_target(self.page, target)
#         if not loc:
#             raise RuntimeError("click: target not found")

#         # Try normal click first; if overlay intercepts, auto-recover.
#         try:
#             loc.click(timeout=30000)
#         except PlaywrightTimeoutError as e:
#             msg = str(e)
#             # Common cross-app pattern: invisible overlay intercepts pointer events
#             if "intercepts pointer events" in msg or "Timeout" in msg:
#                 try:
#                     # try to dismiss overlays/popovers generically
#                     self.page.keyboard.press("Escape")
#                     self.page.wait_for_timeout(150)  # tiny pause
#                     loc.click(timeout=10000)
#                 except Exception:
#                     # last resort: force click
#                     loc.click(timeout=8000, force=True)
#             else:
#                 raise
#         except Exception:
#             # Last resort
#             loc.click(force=True)

#         return self._capture(f"click:{_short_target(target)}")






#     def fill_flexible(self, target: Optional["Target"], text: str):
#         """
#         Title + Numbered List behavior for Notion-like editors:
#         - If multiline: first line becomes the title, then we move into body,
#           enable numbered list, and insert each remaining line as an item.
#         - If single line: just insert that text (no list toggling).
#         """
#         loc = resolve_target(self.page, target) if target else None
#         raw = (text or "").replace("\r\n", "\n").strip("\n")

#         try:
#             # Focus target or any editor
#             if loc:
#                 try:
#                     loc.click(timeout=30000)
#                 except PlaywrightTimeoutError:
#                     self.page.keyboard.press("Escape")
#                     self.page.wait_for_timeout(150)
#                     loc.click(timeout=10000)
#             else:
#                 loc = self.page.locator('[contenteditable="true"]').first
#                 loc.wait_for(timeout=2000)
#                 loc.click()

#             # Clear existing content
#             _clear_editable(self, loc)

#             # Helpers to detect title-like block
#             def _is_contenteditable(l) -> bool:
#                 try:
#                     return bool(l.evaluate("el => el?.getAttribute?.('contenteditable') === 'true'"))
#                 except Exception:
#                     return False
            
#             def _press_any(chords: list[str]) -> None:
#                 """Press the first chord that works on this OS (e.g., Meta+Shift+7 or Control+Shift+7)."""
#                 for chord in chords:
#                     try:
#                         self.page.keyboard.press(chord)
#                         return
#                     except Exception:
#                         continue

#             def _looks_like_title(l) -> bool:
#                 try:
#                     return bool(l.evaluate("""
#                         el => {
#                           const get = k => (el.getAttribute?.(k) || '').toLowerCase();
#                           const ph = get('placeholder');
#                           const role = (el.getAttribute?.('role') || '').toLowerCase();
#                           const tag  = (el.tagName || '').toLowerCase();
#                           const aria = (el.getAttribute?.('aria-label') || '').toLowerCase();
#                           return ph.includes('untitled') || role === 'heading' || tag === 'h1' || aria.includes('title');
#                         }
#                     """))
#                 except Exception:
#                     return False

#             is_ce = _is_contenteditable(loc)
#             is_title = _looks_like_title(loc)

#             # Split lines; first line becomes title if we are on a title block
#             lines = [ln.rstrip() for ln in raw.split("\n")]
#             lines = [ln for ln in lines if ln != ""]  # drop empty-only lines

#             if not lines:
#                 return self._capture("fill:(empty)")

#             if is_ce and is_title and len(lines) > 1:
#                 # ---- Title ----
#                 title = lines[0].strip()
#                 if title:
#                     self.page.keyboard.insert_text(title)

#                 # Move into the body; two Enters is robust in Notion
#                 self.page.keyboard.press("Enter")
#                 self.page.keyboard.press("Enter")

#                 body_items = lines[1:]

#                 # ---- Enable Notion Numbered List ----
#                 # Mac:  Meta+Shift+7 ; Win/Linux: Control+Shift+7
#                 _press_any(["Meta+Shift+7", "Control+Shift+7"])

#                 # Insert items; strip any leading N. prefixes to avoid double numbering
#                 # import re as _re_local
#                 for i, ln in enumerate(body_items):
#                     clean = re.sub(r"^\s*\d+\.\s*", "", ln).strip()
#                     if clean:
#                         self.page.keyboard.insert_text(clean)
#                     if i < len(body_items) - 1:
#                         self.page.keyboard.press("Enter")

#             else:
#                 # Single line or not on a title block: insert atomically
#                 # If multiline here, we DON'T toggle numbered list automatically
#                 # (keeps generic behavior for non-title targets)
#                 if len(lines) == 1:
#                     self.page.keyboard.insert_text(lines[0])
#                 else:
#                     self.page.keyboard.insert_text("\n".join(lines))

#         except PlaywrightTimeoutError as e:
#             print(f"[WARN] fill timeout: {e}")
#         except Exception as e:
#             print(f"[ERROR] fill failed: {e}")
#             raise

#         return self._capture(f"fill:{(raw or '')[:20]}")
class BrowserEnv:
    def __init__(
        self,
        headless: bool = True,
        dataset_root: str = "datasets",
        app_name: Optional[str] = None,
        task_title: Optional[str] = None,
        task_blurb: Optional[str] = None,
    ):
        self.headless = headless
        self.dataset_root = Path(dataset_root)

        # Dataset organization
        self.app_name = slugify(app_name or "unknown-app")
        self.task_title = task_title or "Untitled Task"
        self.task_slug = slugify(self.task_title)
        self.task_blurb = task_blurb or f"Captured UI states for task: {self.task_title}"

        self.playwright = None
        self.browser = None
        self.page = None

        # Derived at runtime
        self.task_dir: Optional[Path] = None
        self.step_counter: int = 0
        self.manifest: List[Dict[str, Any]] = []

    def __enter__(self):
        self.playwright = sync_playwright().start()
        user_data_dir = os.path.abspath("playwright_data")
        self.browser = self.playwright.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=self.headless,
        )
        pages = self.browser.pages
        self.page = pages[0] if pages else self.browser.new_page()
        self.page.set_viewport_size({"width": 1440, "height": 900})

        # datasets/<app>/<task>/
        self.task_dir = self.dataset_root / self.app_name / self.task_slug
        self.task_dir.mkdir(parents=True, exist_ok=True)

        # README for this task (short blurb)
        readme = self.task_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                (
                    f"# {self.task_title}\n\n"
                    f"{self.task_blurb}\n\n"
                    "## Dataset structure\n"
                    "- `manifest.json` – metadata for each captured UI state in order.\n"
                    "- `NNN_<label>.png` – screenshots for each step.\n"
                ),
                encoding="utf-8",
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Dump manifest (all steps in order)
        try:
            if self.task_dir is not None:
                (self.task_dir / "manifest.json").write_text(
                    json.dumps(self.manifest, indent=2),
                    encoding="utf-8",
                )
        except Exception as e:
            print(f"[WARN] failed writing manifest.json: {e}")

        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def press_keys(self, sequence: str):
        """Press keys or type text. Supports comma-separated sequences."""
        if not sequence:
            sequence = "Enter"
        parts = [p.strip() for p in sequence.split(",") if p.strip()]
        for p in parts:
            try:
                # If it looks like a key or chord, press; otherwise type text
                if len(p) == 1 or "+" in p or p.lower() in {
                    "enter","escape","esc","tab","shift","control","ctrl","alt","meta","command","cmd",
                    "arrowup","arrowdown","arrowleft","arrowright","home","end","pageup","pagedown",
                    "backspace","delete","space"
                }:
                    self.page.keyboard.press(p)
                else:
                    self.page.keyboard.type(p)
            except Exception:
                self.page.keyboard.type(p)
        return self._capture(f"press:{sequence}")

    def _capture(self, label: str) -> Dict[str, Any]:
        """
        Capture current UI state:
        - Screenshot saved under datasets/<app>/<task>/NNN_<label>.png
        - Text snippet, affordances, URL, hash, timestamp
        - Append to in-memory manifest (written to manifest.json on __exit__)
        """
        ts = int(time.time() * 1000)

        # Text
        try:
            text = self.page.inner_text("body")
        except Exception:
            text = ""

        # URL + state hash
        url = self.page.url if self.page else ""
        state_hash = (
            hashlib.sha256((url + text[:2000]).encode("utf-8")).hexdigest()
            if url else ""
        )

        # Step index + filename
        self.step_counter += 1
        safe_label = _sanitize_label(label, max_len=40)
        fname = f"{self.step_counter:03d}_{safe_label}.png"

        screenshot_path = ""
        try:
            # Prefer task_dir; fallback to flat dataset_root if missing
            base_dir = self.task_dir or self.dataset_root
            base_dir.mkdir(parents=True, exist_ok=True)
            outpath = base_dir / fname
            if self.page:
                self.page.screenshot(path=str(outpath), full_page=True)
            screenshot_path = str(outpath)
        except Exception as e:
            print(f"[ERROR] screenshot failed: {e}")

        # Affordances (best-effort)
        try:
            affordances = _collect_affordances(self.page, limit=40)
        except Exception:
            affordances = []

        obs = {
            "step_index": self.step_counter,
            "timestamp_ms": ts,
            "url": url,
            "label": label,
            "screenshot_path": screenshot_path,
            "text_snippet": text[:800],
            "affordances": affordances,
            "state_hash": state_hash,
        }

        # Append to manifest (keeps tasks ordered)
        self.manifest.append(obs)
        return obs

    def wait_for_flexible(self, selector: Optional[str] = None, timeout: int = 15000):
        """
        Wait for a selector or general page readiness after an action.
        - If selector is given: waits for it explicitly.
        - If no selector: waits for network idle + main contenteditable.
        """
        try:
            if selector:
                self.page.wait_for_selector(selector, timeout=timeout, state="visible")
            else:
                self.page.wait_for_load_state("networkidle", timeout=timeout)
                try:
                    self.page.locator('[contenteditable="true"]').first.wait_for(timeout=timeout)
                except Exception:
                    self.page.wait_for_selector("body", timeout=timeout, state="visible")
            return True
        except Exception as e:
            print(f"[WARN] wait_for_flexible failed: {e}")
            return False

    def goto(self, url: str, wait_until: str = "domcontentloaded", timeout: int = 45000):
        try:
            self.page.goto(url, wait_until=wait_until, timeout=timeout)
        except PlaywrightTimeoutError as e:
            print(f"[WARN] goto timeout for {url}: {e}")
            try:
                self.page.wait_for_load_state("domcontentloaded", timeout=5000)
            except Exception:
                pass
        except Exception as e:
            print(f"[ERROR] goto failed for {url}: {e}")
        return self._capture(f"goto:{url}")

    def click_flexible(self, target: "Target"):
        loc = resolve_target(self.page, target)
        if not loc:
            raise RuntimeError("click: target not found")

        try:
            loc.click(timeout=30000)
        except PlaywrightTimeoutError as e:
            msg = str(e)
            if "intercepts pointer events" in msg or "Timeout" in msg:
                try:
                    self.page.keyboard.press("Escape")
                    self.page.wait_for_timeout(150)
                    loc.click(timeout=10000)
                except Exception:
                    loc.click(timeout=8000, force=True)
            else:
                raise
        except Exception:
            loc.click(force=True)

        return self._capture(f"click:{_short_target(target)}")

    def fill_flexible(self, target: Optional["Target"], text: str):
        """
        Generic 'fill' that also supports title + numbered-list behavior
        in rich text editors. Still works fine for non-Notion apps.
        """
        loc = resolve_target(self.page, target) if target else None
        raw = (text or "").replace("\r\n", "\n").strip("\n")

        try:
            if loc:
                try:
                    loc.click(timeout=30000)
                except PlaywrightTimeoutError:
                    self.page.keyboard.press("Escape")
                    self.page.wait_for_timeout(150)
                    loc.click(timeout=10000)
            else:
                loc = self.page.locator('[contenteditable="true"]').first
                loc.wait_for(timeout=2000)
                loc.click()

            _clear_editable(self, loc)

            def _is_contenteditable(l) -> bool:
                try:
                    return bool(l.evaluate("el => el?.getAttribute?.('contenteditable') === 'true'"))
                except Exception:
                    return False

            def _press_any(chords: list[str]) -> None:
                for chord in chords:
                    try:
                        self.page.keyboard.press(chord)
                        return
                    except Exception:
                        continue

            def _looks_like_title(l) -> bool:
                try:
                    return bool(l.evaluate("""
                        el => {
                          const get = k => (el.getAttribute?.(k) || '').toLowerCase();
                          const ph = get('placeholder');
                          const role = (el.getAttribute?.('role') || '').toLowerCase();
                          const tag  = (el.tagName || '').toLowerCase();
                          const aria = (el.getAttribute?.('aria-label') || '').toLowerCase();
                          return ph.includes('untitled') || role == 'heading' || tag == 'h1' || aria.includes('title');
                        }
                    """))
                except Exception:
                    return False

            is_ce = _is_contenteditable(loc)
            is_title = _looks_like_title(loc)

            lines = [ln.rstrip() for ln in raw.split("\n")]
            lines = [ln for ln in lines if ln != ""]

            if not lines:
                return self._capture("fill:(empty)")

            if is_ce and is_title and len(lines) > 1:
                title = lines[0].strip()
                if title:
                    self.page.keyboard.insert_text(title)

                self.page.keyboard.press("Enter")
                self.page.keyboard.press("Enter")

                body_items = lines[1:]

                _press_any(["Meta+Shift+7", "Control+Shift+7"])

                for i, ln in enumerate(body_items):
                    clean = re.sub(r"^\s*\d+\.\s*", "", ln).strip()
                    if clean:
                        self.page.keyboard.insert_text(clean)
                    if i < len(body_items) - 1:
                        self.page.keyboard.press("Enter")
            else:
                if len(lines) == 1:
                    self.page.keyboard.insert_text(lines[0])
                else:
                    self.page.keyboard.insert_text("\n".join(lines))

        except PlaywrightTimeoutError as e:
            print(f"[WARN] fill timeout: {e}")
        except Exception as e:
            print(f"[ERROR] fill failed: {e}")
            raise

        return self._capture(f"fill:{(raw or '')[:20]}")


def _short_target(t: Optional[Target]) -> str:
    if not t:
        return "(none)"
    if t.role and t.name:
        return f"role={t.role}[name={t.name[:30]}]"
    if t.text:
        return f"text={t.text[:30]}"
    if t.css:
        return f"css={t.css[:30]}"
    if t.xpath:
        return f"xpath={t.xpath[:30]}"
    return "(target)"


# Global browser handle for nodes
BROWSER: Optional[BrowserEnv] = None


# Shared State
class AgentState(TypedDict, total=False):
    task: str
    steps: List[Dict[str, Any]]          
    done: bool
    next_action: Dict[str, Any]
    history: List[Dict[str, Any]]
    last_state_hash: Optional[str]
    max_steps: int


# LLM + constants
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    response_format={"type": "json_object"},
)
ALLOWED_ACTIONS = {"goto", "click", "fill", "press", "wait_for", "hover", "upload", "finish"}


def _image_to_data_url(path: str) -> Optional[str]:
    try:
        if not path or not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None

_JSON_OBJECT_RE = regex.compile(r"\{(?:[^{}]|(?R))*\}", re.S)  # recursive-ish matcher



# Planner Node (multimodal: text + screenshot + affordances)

def _image_to_data_url(path: str) -> Optional[str]:
    try:
        if not path or not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None

def planner_node(state: AgentState) -> AgentState:
    task = state["task"]
    steps = state.get("steps", [])
    last_obs = steps[-1]["observation"] if steps else None
    max_steps = state.get("max_steps", 10)

    if len(steps) >= max_steps:
        action = PlanStep(
            action="finish",
            description=f"Max steps ({max_steps}) reached. Finishing.",
            done=True,
            save_state=True,
        ).dict()
        print("[planner] Max steps reached, finishing.")
        state["next_action"] = action
        state["done"] = True
        return state

    # Bootstrap
    if not steps:
        CANONICAL_HOSTS = {
            "linear": "https://linear.app",
            "notion": "https://www.notion.so",
            "asana": "https://app.asana.com",
        }

        bootstrap_system = SystemMessage(
            content=(
                "You must return STRICT JSON with keys: url, description. "
                "Pick the best starting URL for the user task. "
                "If unsure, use https://www.google.com."
            )
        )
        bootstrap_user = HumanMessage(
            content=f"Task: {task}\nReturn only JSON: {{\"url\": str, \"description\": str}}"
        )
        res = llm.invoke([bootstrap_system, bootstrap_user])
        print("\n[planner-bootstrap] raw LLM output:\n", res.content)

        start_url = "https://www.google.com"
        desc = "Open https://www.google.com as a generic starting point."

        try:
            meta = json.loads(strip_js_comments(res.content))
            if isinstance(meta, dict):
                u = (meta.get("url") or "").strip()
                d = (meta.get("description") or "").strip()
                if u:
                    start_url = u
                if d:
                    desc = d
        except Exception:
            print("[planner-bootstrap] invalid JSON, using fallback google.com")

        task_lower = task.lower()
        for name, canonical_url in CANONICAL_HOSTS.items():
            if name in task_lower:
                if canonical_url != start_url:
                    print(f"[planner-bootstrap] overriding '{start_url}' → '{canonical_url}' for '{name}'")
                start_url = canonical_url
                desc = f"Open {canonical_url} as starting point for {name}."
                break

        if not start_url.startswith("http"):
            start_url = "https://www.google.com"
            desc = "Open https://www.google.com as a safe starting point."

        first_action = PlanStep(
            action="goto",
            url=start_url,
            description=desc,
            save_state=True,
        ).dict()

        print("[planner] bootstrap first action:", first_action)
        state["next_action"] = first_action
        return state

    # Normal planning (now with screenshot)
    if last_obs:
        affordances = last_obs.get("affordances", [])
        ui_text = (
            f"Last URL: {last_obs.get('url','')}\n"
            f"Last label: {last_obs.get('label','')}\n"
            f"Visible text (trunc):\n{last_obs.get('text_snippet','')}\n\n"
            f"Affordances (clickable/menu-ish items): {affordances}"
        )
        img_url = _image_to_data_url(last_obs.get("screenshot_path"))
        # Build a multimodal user message
        user_msg_content = [{"type": "text", "text": (
            f"Task: {task}\n\n"
            f"Recent steps (truncated): {json.dumps(steps[-4:], indent=2)}\n\n"
            f"Current UI (text + affordances):\n{ui_text}\n\n"
            "Decide the next best atomic action."
        )}]
        if img_url:
            user_msg_content.append({"type": "image_url", "image_url": {"url": img_url}})
    else:
        # No observation; send text-only context
        user_msg_content = [{"type": "text", "text": (
            f"Task: {task}\n\n"
            f"Recent steps (truncated): {json.dumps(steps[-4:], indent=2)}\n\n"
            "No valid UI snapshot available. Decide the next best atomic action."
        )}]

    system = SystemMessage(
        content=(
            "Return ONE atomic action as STRICT JSON matching:\n"
            "{"
            '  \"action\": \"goto|click|fill|press|wait_for|hover|upload|finish\",'
            '  \"target\": { \"role\"?: str, \"name\"?: str, \"text\"?: str, \"css\"?: str, \"xpath\"?: str, \"match\"?: \"auto|exact|contains|regex\" },'
            '  \"value\"?: str, \"url\"?: str, \"save_state\": bool, \"description\": str, \"done\": bool'
            "}\n"
            "Rules:\n"
            "- Your actions must align with the user's requested object type.\n"
            "  • If the user asks for a *page* or *document*, avoid creating or editing objects clearly labeled as "
            "    'database', 'table', 'board', 'project', etc., unless the task explicitly mentions them.\n"
            "- Prefer role+name; else text; else css/xpath.\n"
            "- If prior click didn't change the UI, prefer an alternative target.\n"
            "- For typing into editors, use role=textbox (the executor supports contenteditable).\n"
            "- For menus or command palettes, you may use action='press' with value like 'Control+K, ... , Enter'.\n"
            "- No comments. No trailing commas. Only the JSON object."
        )
    )


    try:
        step = plan_or_retry(llm, [system, HumanMessage(content=user_msg_content)], tries=2)
        action = step.dict()
    except Exception as e:
        print(f"[planner] Plan parse failed: {e}")
        action = PlanStep(
            action="finish",
            description="Planner returned invalid JSON repeatedly.",
            done=True,
            save_state=True,
        ).dict()

    if action["action"] not in ALLOWED_ACTIONS:
        action = PlanStep(
            action="finish",
            description="Unsupported action from planner.",
            done=True,
            save_state=True,
        ).dict()

    print("[planner] parsed action:", action)
    state["next_action"] = action
    state.setdefault("history", []).append({"role": "planner", "raw": json.dumps(action)})
    if action.get("done") or action.get("action") == "finish":
        state["done"] = True
    return state


# def planner_node(state: AgentState) -> AgentState:
#     task = state["task"]
#     steps = state.get("steps", [])
#     last_obs = steps[-1]["observation"] if steps else None
#     max_steps = state.get("max_steps", 15)

#     if len(steps) >= max_steps:
#         action = PlanStep(
#             action="finish",
#             description=f"Max steps ({max_steps}) reached. Finishing.",
#             done=True,
#             save_state=True,
#         ).dict()
#         print("[planner] Max steps reached, finishing.")
#         state["next_action"] = action
#         state["done"] = True
#         return state

#     # Bootstrap
#     if not steps:
#         CANONICAL_HOSTS = {
#             "linear": "https://linear.app",
#             "notion": "https://www.notion.so",
#             "asana": "https://app.asana.com",
#         }

#         bootstrap_system = SystemMessage(
#             content=(
#                 "You must return STRICT JSON with keys: url, description. "
#                 "Pick the best starting URL for the user task. "
#                 "If unsure, use https://www.google.com."
#             )
#         )
#         bootstrap_user = HumanMessage(
#             content=f"Task: {task}\nReturn only JSON: {{\"url\": str, \"description\": str}}"
#         )
#         res = llm.invoke([bootstrap_system, bootstrap_user])
#         print("\n[planner-bootstrap] raw LLM output:\n", res.content)

#         start_url = "https://www.google.com"
#         desc = "Open https://www.google.com as a generic starting point."

#         try:
#             meta = json.loads(strip_js_comments(res.content))
#             if isinstance(meta, dict):
#                 u = (meta.get("url") or "").strip()
#                 d = (meta.get("description") or "").strip()
#                 if u:
#                     start_url = u
#                 if d:
#                     desc = d
#         except Exception:
#             print("[planner-bootstrap] invalid JSON, using fallback google.com")

#         task_lower = task.lower()
#         for name, canonical_url in CANONICAL_HOSTS.items():
#             if name in task_lower:
#                 if canonical_url != start_url:
#                     print(f"[planner-bootstrap] overriding '{start_url}' → '{canonical_url}' for '{name}'")
#                 start_url = canonical_url
#                 desc = f"Open {canonical_url} as starting point for {name}."
#                 break

#         if not start_url.startswith("http"):
#             start_url = "https://www.google.com"
#             desc = "Open https://www.google.com as a safe starting point."

#         first_action = PlanStep(
#             action="goto",
#             url=start_url,
#             description=desc,
#             save_state=True,
#         ).dict()

#         print("[planner] bootstrap first action:", first_action)
#         state["next_action"] = first_action
#         return state

#     # Normal planning
#     if last_obs:
#         ui_context = (
#             f"Last URL: {last_obs.get('url','')}\n"
#             f"Last label: {last_obs.get('label','')}\n"
#             f"Visible text:\n{last_obs.get('text_snippet','')}"
#         )
#     else:
#         ui_context = "No valid UI snapshot."

#     system = SystemMessage(
#         content=(
#             "Return ONE atomic action as STRICT JSON matching:\n"
#             "{"
#             '  "action": "goto|click|fill|press|wait_for|hover|upload|finish",'
#             '  "target": { "role"?: str, "name"?: str, "text"?: str, "css"?: str, "xpath"?: str, "match"?: "auto|exact|contains|regex" },'
#             '  "value"?: str, "url"?: str, "save_state": bool, "description": str, "done": bool'
#             "}\n"
#             "Rules:\n"
#             "- Prefer role+name; else text; else css/xpath.\n"
#             "- If prior click didn't change the UI, prefer an alternative target.\n"
#             "- For typing into editors, use role=textbox (the executor supports contenteditable).\n"
#             "- For menus or command palettes, you may use action='press' with value like 'Control+K, ... , Enter'.\n"
#             "- No comments. No trailing commas. Only the JSON object."
#         )
#     )

#     user = HumanMessage(
#         content=(
#             f"Task: {task}\n\n"
#             f"Recent steps (truncated): {json.dumps(steps[-4:], indent=2)}\n\n"
#             f"Current UI:\n{ui_context}\n\n"
#             "Decide the next best atomic action."
#         )
#     )

#     try:
#         step = plan_or_retry(llm, [system, user], tries=2)
#         action = step.dict()
#     except Exception as e:
#         print(f"[planner] Plan parse failed: {e}")
#         action = PlanStep(
#             action="finish",
#             description="Planner returned invalid JSON repeatedly.",
#             done=True,
#             save_state=True,
#         ).dict()

#     # Ensure allowed
#     if action["action"] not in ALLOWED_ACTIONS:
#         action = PlanStep(
#             action="finish",
#             description="Unsupported action from planner.",
#             done=True,
#             save_state=True,
#         ).dict()

#     print("[planner] parsed action:", action)
#     state["next_action"] = action
#     state.setdefault("history", []).append({"role": "planner", "raw": json.dumps(action)})
#     if action.get("done") or action.get("action") == "finish":
#         state["done"] = True
#     return state


def _record_step(state: AgentState, action: Dict[str, Any], obs: Dict[str, Any], *, force: bool = False):
    steps = state.get("steps", [])
    last_hash = state.get("last_state_hash")
    save_state = bool(action.get("save_state", False))
    h = obs.get("state_hash")
    is_new_state = bool(h and h != last_hash)

    should_record = force or save_state or is_new_state or obs.get("label","").startswith("error:")
    if should_record:
        steps.append({"action": action, "observation": {k: v for k, v in obs.items()}})
        state["steps"] = steps
        state["last_state_hash"] = h or last_hash
        print("[executor] recorded step. total steps:", len(steps))
    else:
        print("[executor] state not recorded (no save_state and no new hash).")



# Executor Node

def executor_node(state: AgentState) -> AgentState:
    global BROWSER
    if BROWSER is None:
        raise RuntimeError("BROWSER is not initialized. Use run_task().")

    # Read action; support legacy shape and coerce to PlanStep
    raw = state.get("next_action") or {}
    try:
        if "selector" in raw and "target" not in raw:
            raw["target"] = selector_string_to_target(str(raw["selector"]))
        step = PlanStep.model_validate(raw)
    except Exception as e:
        print(f"[executor] invalid action in state: {e}")
        state["done"] = True
        return state

    kind = step.action
    if kind == "finish":
        state["done"] = True
        return state

    print(f"[executor] executing: {step.dict()}")

    # First-time manual login guard per host
    if kind == "goto":
        target_url = step.url or ""
        host = _host_from_url(target_url)
        if host and not _has_manual_login(host):
            if BROWSER.headless:
                raise RuntimeError(
                    f"[login-guard] First-time login required for '{host}'. "
                    "Run headless=False once to log in manually."
                )
            obs = BROWSER.goto(target_url)
            print(
                f"\n[login-guard] First visit to '{host}'.\n"
                "Please complete login manually in the visible browser.\n"
                "Return here and press ENTER when done."
            )
            try:
                input("[login-guard] Press ENTER to continue after login...")
            except EOFError:
                BROWSER.page.wait_for_timeout(3000)
            _mark_manual_login_done(host)
            # Capture a post-login state
            obs = BROWSER._capture(f"post-login:{host}")
            _record_step(state, step.dict(), obs)
            state["next_action"] = {}
            return state

    # Normal execution
   # --- Normal execution ---
    try:
        if kind == "goto":
            obs = BROWSER.goto(step.url or "")
        elif kind == "click":
            obs = BROWSER.click_flexible(step.target)
        elif kind == "fill":
            obs = BROWSER.fill_flexible(step.target, step.value or "")
        elif kind == "press":
            obs = BROWSER.press_keys(step.value or "Enter")
        elif kind == "wait_for":
            BROWSER.wait_for_flexible()
            obs = BROWSER._capture(f"wait_for:{_short_target(step.target)}")
        elif kind == "hover":
            loc = resolve_target(BROWSER.page, step.target)
            if not loc:
                raise RuntimeError("hover: target not found")
            loc.hover()
            obs = BROWSER._capture(f"hover:{_short_target(step.target)}")
        elif kind == "upload":
            loc = resolve_target(BROWSER.page, step.target)
            if not loc:
                raise RuntimeError("upload: target not found")
            loc.set_input_files(step.value or "")
            obs = BROWSER._capture(f"upload:{_short_target(step.target)}")
        else:
            obs = BROWSER._capture(f"unknown:{kind}")

        # take an extra screenshot after each step, even if unchanged
        extra = BROWSER._capture(f"post-{kind}")
        obs["post_screenshot_path"] = extra["screenshot_path"]


    except Exception as e:
        print(f"[executor] error while executing {kind}: {e}")
        # Take a real screenshot + text + hash so planner can reason
        obs = BROWSER._capture(f"error:{kind}")
        obs["error"] = str(e)

# Always record, so planner gets the latest screenshot each step
    force_record = True
    _record_step(state, step.dict(), obs, force=force_record)
    state["next_action"] = {}
    return state

# Graph

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_edge(START, "planner")

    def route_from_planner(state: AgentState) -> str:
        return END if state.get("done") else "executor"

    graph.add_conditional_edges("planner", route_from_planner, {END: END, "executor": "executor"})
    graph.add_edge("executor", "planner")
    return graph.compile()  # <-- add this

# Orchestrator

def run_task(task: str, headless: bool = True, max_steps: int = 10) -> AgentState:
    global BROWSER
    app = build_graph()

    initial_state: AgentState = {
        "task": task,
        "steps": [],
        "done": False,
        "next_action": {},
        "history": [],
        "last_state_hash": None,
        "max_steps": max_steps,
    }

    app_name = infer_app_name(task)

    with BrowserEnv(
        headless=headless,
        dataset_root="datasets",
        app_name=app_name,
        task_title=task,
        task_blurb=f"Captured UI states for: {task}",
    ) as browser:
        BROWSER = browser
        final_state = app.invoke(initial_state)

    BROWSER = None
    return final_state


# CLI
if __name__ == "__main__":
    print("Enter tasks for Agent B to execute (one per line).")
    print("Examples:")
    print("  Show me how to open Linear.")
    print("  Show me how to open Notion.")
    print("  Start creating a new project in Asana.")
    print("Press ENTER on an empty line when you're done.\n")

    demo_tasks: List[str] = []
    while True:
        try:
            line = input("Task: ").strip()
        except EOFError:
            break
        if not line:
            break
        demo_tasks.append(line)

    if not demo_tasks:
        print("No tasks provided. Exiting.")
        raise SystemExit

    print(f"\nRunning {len(demo_tasks)} task(s)...\n")

    for t in demo_tasks:
        print("\n" + "=" * 80)
        print("TASK:", t)

        # NOTE: First run per host must be headless=False so you can log in manually.
        result = run_task(t, headless=False, max_steps=10)

        steps = result.get("steps", [])
        print(f"\nCaptured {len(steps)} key steps for task: {t!r}")

        if not steps:
            print("No steps captured. Check planner logs above for JSON/action issues.")
            continue

        for i, step in enumerate(steps, start=1):
            action = step["action"]
            obs = step["observation"]
            print(f"\nStep {i}: {action.get('description','(no description)')}")
            print("  Action:      ", action.get("action"))
            if action.get("target"):
                print("  Target:      ", action.get("target"))
            if action.get("url"):
                print("  Target URL:  ", action.get("url"))
            if action.get("value"):
                print("  Value:       ", action.get("value"))
            print("  Page URL:    ", obs.get("url", ""))
            print("  Screenshot:  ", obs.get("screenshot_path", ""))
            print("  Label:       ", obs.get("label", ""))

    print("\nAll tasks finished. Check the 'screenshots/' folder for captured UI states.")
