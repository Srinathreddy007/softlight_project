import os
import time
import json
import hashlib
from typing import TypedDict, List, Dict, Any, Optional

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END


# ========== BrowserEnv (Agent A's tool) ==========

class BrowserEnv:
    def __init__(self, headless: bool = True, screenshot_dir: str = "screenshots"):
        self.headless = headless
        self.screenshot_dir = screenshot_dir
        self.playwright = None
        self.browser = None
        self.page = None

    def __enter__(self):
        self.playwright = sync_playwright().start()

        # Use persistent context so login sessions are reused
        user_data_dir = os.path.abspath("playwright_data")

        self.browser = self.playwright.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=self.headless,
        )

        pages = self.browser.pages
        self.page = pages[0] if pages else self.browser.new_page()
        self.page.set_viewport_size({"width": 1440, "height": 900})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def goto(self, url: str, wait_until: str = "domcontentloaded", timeout: int = 60000):
        try:
            self.page.goto(url, wait_until=wait_until, timeout=timeout)
        except PlaywrightTimeoutError as e:
            print(f"[WARN] goto timeout for {url}: {e}")
        except Exception as e:
            print(f"[ERROR] goto failed for {url}: {e}")
        return self._capture(f"goto:{url}")

    def click(self, selector: str):
        try:
            self.page.click(selector)
        except PlaywrightTimeoutError as e:
            print(f"[WARN] click timeout for {selector}: {e}")
        except Exception as e:
            print(f"[ERROR] click failed for {selector}: {e}")
        return self._capture(f"click:{selector}")

    def fill(self, selector: str, text: str):
        try:
            self.page.fill(selector, text)
        except PlaywrightTimeoutError as e:
            print(f"[WARN] fill timeout for {selector}: {e}")
        except Exception as e:
            print(f"[ERROR] fill failed for {selector}: {e}")
        return self._capture(f"fill:{selector}")

    def press(self, key: str):
        try:
            self.page.keyboard.press(key)
        except Exception as e:
            print(f"[ERROR] press failed for {key}: {e}")
        return self._capture(f"press:{key}")

    def wait_for(self, selector: str, timeout: int = 8000):
        try:
            self.page.wait_for_selector(selector, timeout=timeout)
        except PlaywrightTimeoutError as e:
            print(f"[WARN] wait_for timeout for {selector}: {e}")
        except Exception as e:
            print(f"[ERROR] wait_for failed for {selector}: {e}")
        return self._capture(f"wait_for:{selector}")

    def _capture(self, label: str) -> Dict[str, Any]:
        os.makedirs(self.screenshot_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        safe_label = label.replace(":", "_").replace("/", "_")
        filename = f"{self.screenshot_dir}/{ts}_{safe_label}.png"

        try:
            self.page.screenshot(path=filename, full_page=True)
        except Exception as e:
            print(f"[ERROR] screenshot failed: {e}")
            filename = ""

        try:
            text = self.page.inner_text("body")
        except Exception:
            text = ""

        url = self.page.url if self.page else ""
        state_hash = hashlib.sha256(
            (url + text[:2000]).encode("utf-8")
        ).hexdigest() if url else ""

        return {
            "url": url,
            "label": label,
            "screenshot_path": filename,
            "text_snippet": text[:800],
            "state_hash": state_hash,
        }


# Global browser used inside executor node
BROWSER: Optional[BrowserEnv] = None


# ========== Shared Graph State ==========

class AgentState(TypedDict, total=False):
    task: str
    steps: List[Dict[str, Any]]          # [{action, observation}]
    done: bool
    next_action: Dict[str, Any]
    history: List[Dict[str, Any]]
    last_state_hash: Optional[str]
    max_steps: int


llm = ChatOllama(model="mistral")


ALLOWED_ACTIONS = {"goto", "click", "fill", "press", "wait_for", "finish"}


# ========== Planner Node (Agent B) ==========

# def planner_node(state: AgentState) -> AgentState:
#     task = state["task"]
#     steps = state.get("steps", [])
#     last_obs = steps[-1]["observation"] if steps else None
#     max_steps = state.get("max_steps", 15)

#     # Hard safety: don't loop forever
#     if len(steps) >= max_steps:
#         action = {
#             "action": "finish",
#             "url": "",
#             "selector": "",
#             "text": "",
#             "save_state": True,
#             "description": f"Max steps ({max_steps}) reached. Finishing.",
#             "done": True,
#         }
#         print("[planner] Max steps reached, finishing.")
#         state["next_action"] = action
#         state["done"] = True
#         return state

#     # Deterministic bootstrap for first step: ensures we test pipeline
#     if not steps:
#         first_action = {
#             "action": "goto",
#             "url": "https://www.notion.so",
#             "selector": "",
#             "text": "",
#             "save_state": True,  # force capture
#             "description": "Open Notion homepage as starting point.",
#             "done": False,
#         }
#         print("[planner] bootstrap first action:", first_action)
#         state["next_action"] = first_action
#         return state

#     if last_obs:
#         ui_context = (
#             f"Last URL: {last_obs.get('url','')}\n"
#             f"Last label: {last_obs.get('label','')}\n"
#             f"Visible text:\n{last_obs.get('text_snippet','')}"
#         )
#     else:
#         ui_context = (
#             "No UI yet. Start with a 'goto' to a reasonable URL based on the task."
#         )

#     system = SystemMessage(
#         content=(
#             "You are Agent B, planning how to operate arbitrary web apps via Agent A.\n"
#             "Emit ONE atomic action as STRICT JSON.\n\n"
#             "Allowed actions:\n"
#             "  - 'goto'     : navigate to URL (set 'url').\n"
#             "  - 'click'    : click element (set 'selector').\n"
#             "  - 'fill'     : type (set 'selector', 'text').\n"
#             "  - 'press'    : keypress (set 'text', e.g. 'Enter').\n"
#             "  - 'wait_for' : wait for selector (set 'selector').\n"
#             "  - 'finish'   : when the tutorial is complete.\n\n"
#             "Return ONLY JSON with keys:\n"
#             "  action, url, selector, text, save_state, description, done\n"
#             "No extra text. If unsure, choose a sensible next UI action."
#         )
#     )

#     user = HumanMessage(
#         content=(
#             f"Task: {task}\n\n"
#             f"Recent steps: {json.dumps(steps[-4:], indent=2)}\n\n"
#             f"Current UI:\n{ui_context}\n"
#             "Decide the next atomic action."
#         )
#     )

#     res = llm.invoke([system, user])

#     print("\n[planner] raw LLM output:")
#     print(res.content)

#     # Try to parse JSON
#     try:
#         action = json.loads(res.content)
#     except Exception:
#         print("[planner] Invalid JSON, forcing finish.")
#         action = {
#             "action": "finish",
#             "url": "",
#             "selector": "",
#             "text": "",
#             "save_state": True,
#             "description": f"Invalid JSON from LLM.",
#             "done": True,
#         }

#     # Normalize + guard rails
#     action.setdefault("url", "")
#     action.setdefault("selector", "")
#     action.setdefault("text", "")
#     action.setdefault("save_state", False)
#     action.setdefault("description", "")
#     action.setdefault("done", False)

#     if action["action"] not in ALLOWED_ACTIONS:
#         print(f"[planner] Invalid action '{action['action']}', forcing finish.")
#         action = {
#             "action": "finish",
#             "url": "",
#             "selector": "",
#             "text": "",
#             "save_state": True,
#             "description": "LLM returned unsupported action.",
#             "done": True,
#         }

#     print("[planner] parsed action:", action)

#     state["next_action"] = action
#     state.setdefault("history", []).append({"role": "planner", "raw": res.content})

#     if action.get("done") or action.get("action") == "finish":
#         state["done"] = True

#     return state

# def planner_node(state: AgentState) -> AgentState:
#     """
#     Agent B:
#     - If no steps yet: ask LLM for a good starting URL based on the task (bootstrap).
#     - Otherwise: read task + recent steps + last UI snapshot, emit ONE atomic action.
#     - Writes result into state["next_action"], may set state["done"].
#     """
#     task = state["task"]
#     steps = state.get("steps", [])
#     last_obs = steps[-1]["observation"] if steps else None
#     max_steps = state.get("max_steps", 15)

#     # 1) Safety: stop if too many steps
#     if len(steps) >= max_steps:
#         action = {
#             "action": "finish",
#             "url": "",
#             "selector": "",
#             "text": "",
#             "save_state": True,
#             "description": f"Max steps ({max_steps}) reached. Finishing.",
#             "done": True,
#         }
#         print("[planner] Max steps reached, finishing.")
#         state["next_action"] = action
#         state["done"] = True
#         return state

#     # 2) Bootstrap: first step = choose start URL via LLM (no hardcoded rules)
#     if not steps:
#         bootstrap_system = SystemMessage(
#             content=(
#                 "You help choose the best starting URL for a web task.\n"
#                 "Given a natural language task, identify which web app it refers to "
#                 "(e.g., Linear, Notion, Asana, Jira, GitHub, etc.) and output a suitable "
#                 "login or workspace/start URL.\n"
#                 "Return ONLY JSON with keys:\n"
#                 "{ \"url\": str, \"description\": str }\n"
#                 "If unsure, choose a reasonable generic starting point (e.g. https://www.google.com)."
#             )
#         )
#         bootstrap_user = HumanMessage(
#             content=f"Task: {task}\nDecide the most appropriate starting URL for this task."
#         )

#         res = llm.invoke([bootstrap_system, bootstrap_user])
#         print("\n[planner-bootstrap] raw LLM output:")
#         print(res.content)

#         try:
#             meta = json.loads(res.content)
#             start_url = meta.get("url", "").strip() or "https://www.google.com"
#             desc = meta.get("description", "").strip() or f"Open {start_url} as starting point."
#         except Exception:
#             print("[planner-bootstrap] invalid JSON, using fallback start URL.")
#             start_url = "https://www.google.com"
#             desc = f"Open {start_url} as a generic starting point."

#         CANONICAL_HOSTS = {
#         "linear": "https://linear.app",
#         "notion": "https://www.notion.so",
#         "asana": "https://app.asana.com",
#         }

#         task_lower = task.lower()
#         for name, canonical_url in CANONICAL_HOSTS.items():
#             if name in task_lower:
#                 print(f"[planner-bootstrap] overriding LLM URL '{start_url}' "
#                       f"with canonical '{canonical_url}' for '{name}'.")
#                 start_url = canonical_url
#                 break

#         first_action = {
#             "action": "goto",
#             "url": start_url,
#             "selector": "",
#             "text": "",
#             "save_state": True,  # always capture initial state
#             "description": desc,
#             "done": False,
#         }

#         print("[planner] bootstrap first action:", first_action)
#         state["next_action"] = first_action
#         return state

#     # 3) Normal planning step (we already have at least one observation)
#     if last_obs:
#         ui_context = (
#             f"Last URL: {last_obs.get('url','')}\n"
#             f"Last label: {last_obs.get('label','')}\n"
#             f"Visible text:\n{last_obs.get('text_snippet','')}"
#         )
#     else:
#         ui_context = "No valid UI snapshot found. Propose a 'goto' or sensible recovery step."

#     system = SystemMessage(
#         content=(
#             "You are Agent B, planning how to operate arbitrary web apps via Agent A.\n"
#             "Emit ONE atomic action as STRICT JSON.\n\n"
#             "Allowed actions:\n"
#             "  - 'goto'     : navigate to URL (set 'url').\n"
#             "  - 'click'    : click element (set 'selector').\n"
#             "  - 'fill'     : type into input (set 'selector', 'text').\n"
#             "  - 'press'    : keypress (set 'text', e.g. 'Enter').\n"
#             "  - 'wait_for' : wait for selector (set 'selector').\n"
#             "  - 'finish'   : when the requested workflow is clearly demonstrated.\n\n"
#             "You are building a tutorial. Set 'save_state': true for important states:\n"
#             "- starting page for the task\n"
#             "- before opening a key modal\n"
#             "- modal / form visible\n"
#             "- form filled before submit\n"
#             "- success / confirmation screen\n\n"
#             "Return ONLY JSON with keys:\n"
#             "  action, url, selector, text, save_state, description, done\n"
#             "No extra keys. No prose. No markdown."
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

#     res = llm.invoke([system, user])

#     print("\n[planner] raw LLM output:")
#     print(res.content)

#     # 4) Parse + normalize
#     try:
#         action = json.loads(res.content)
#     except Exception:
#         print("[planner] Invalid JSON, forcing finish.")
#         action = {
#             "action": "finish",
#             "url": "",
#             "selector": "",
#             "text": "",
#             "save_state": True,
#             "description": "Invalid JSON from LLM.",
#             "done": True,
#         }

#     # Ensure all expected keys exist
#     action.setdefault("url", "")
#     action.setdefault("selector", "")
#     action.setdefault("text", "")
#     action.setdefault("save_state", False)
#     action.setdefault("description", "")
#     action.setdefault("done", False)

#     # Guardrail on allowed actions
#     if action["action"] not in {"goto", "click", "fill", "press", "wait_for", "finish"}:
#         print(f"[planner] Unsupported action '{action['action']}', forcing finish.")
#         action = {
#             "action": "finish",
#             "url": "",
#             "selector": "",
#             "text": "",
#             "save_state": True,
#             "description": "LLM returned unsupported action.",
#             "done": True,
#         }

#     print("[planner] parsed action:", action)

#     state["next_action"] = action
#     state.setdefault("history", []).append({"role": "planner", "raw": res.content})

#     if action.get("done") or action.get("action") == "finish":
#         state["done"] = True

#     return state
def planner_node(state: AgentState) -> AgentState:
    """
    Agent B:
    - If no steps yet: choose a starting URL (LLM + small canonical map).
    - Otherwise: based on task + recent steps + last UI, emit ONE atomic action.
    - Writes result into state["next_action"], may set state["done"].
    """
    task = state["task"]
    steps = state.get("steps", [])
    last_obs = steps[-1]["observation"] if steps else None
    max_steps = state.get("max_steps", 15)

    # ---------- 0. Safety: bail if too many steps ----------
    if len(steps) >= max_steps:
        action = {
            "action": "finish",
            "url": "",
            "selector": "",
            "text": "",
            "save_state": True,
            "description": f"Max steps ({max_steps}) reached. Finishing.",
            "done": True,
        }
        print("[planner] Max steps reached, finishing.")
        state["next_action"] = action
        state["done"] = True
        return state

    # ---------- 1. Bootstrap: first step ----------
    if not steps:
        CANONICAL_HOSTS = {
            "linear": "https://linear.app",
            "notion": "https://www.notion.so",
            "asana": "https://app.asana.com",
        }

        bootstrap_system = SystemMessage(
            content=(
                "You help choose the best starting URL for a web task.\n"
                "Given a natural language task, decide which web app it refers to "
                "and output ONLY JSON: { \"url\": str, \"description\": str }.\n"
                "If unsure, use https://www.google.com."
            )
        )
        bootstrap_user = HumanMessage(
            content=f"Task: {task}\nDecide the most appropriate starting URL for this task."
        )

        res = llm.invoke([bootstrap_system, bootstrap_user])
        print("\n[planner-bootstrap] raw LLM output:")
        print(res.content)

        # Defaults
        start_url = "https://www.google.com"
        desc = "Open https://www.google.com as a generic starting point."

        # Parse LLM answer
        try:
            meta = json.loads(res.content)
            if isinstance(meta, dict):
                url_val = (meta.get("url") or "").strip()
                desc_val = (meta.get("description") or "").strip()
                if url_val:
                    start_url = url_val
                if desc_val:
                    desc = desc_val
        except Exception:
            print("[planner-bootstrap] invalid JSON, using fallback google.com")

        # Canonical override for known tools
        task_lower = task.lower()
        for name, canonical_url in CANONICAL_HOSTS.items():
            if name in task_lower:
                if canonical_url != start_url:
                    print(
                        f"[planner-bootstrap] overriding LLM URL '{start_url}' "
                        f"with canonical '{canonical_url}' for '{name}'."
                    )
                start_url = canonical_url
                if not desc or "generic" in desc.lower():
                    desc = f"Open {canonical_url} as starting point for {name}."
                break

        # Ensure valid URL
        if not start_url.startswith("http"):
            print(
                f"[planner-bootstrap] non-HTTP URL from LLM: '{start_url}', "
                "using https://www.google.com instead."
            )
            start_url = "https://www.google.com"
            desc = "Open https://www.google.com as a safe starting point."

        first_action = {
            "action": "goto",
            "url": start_url,
            "selector": "",
            "text": "",
            "save_state": True,
            "description": desc,
            "done": False,
        }

        print("[planner] bootstrap first action:", first_action)
        state["next_action"] = first_action
        return state

    # ---------- 2. Normal planning (subsequent steps) ----------

    if last_obs:
        ui_context = (
            f"Last URL: {last_obs.get('url','')}\n"
            f"Last label: {last_obs.get('label','')}\n"
            f"Visible text:\n{last_obs.get('text_snippet','')}"
        )
    else:
        ui_context = "No valid UI snapshot found. Suggest a 'goto' or recovery action."

    system = SystemMessage(
        content=(
        "You are Agent B, planning how to operate arbitrary web apps via Agent A.\n"
        "Emit ONE atomic action as STRICT JSON.\n"
        "Allowed actions: 'goto', 'click', 'fill', 'press', 'wait_for', 'finish'.\n"
        "Selectors MUST be robust:\n"
        "- Prefer Playwright text selectors: e.g. \"text=Log in\", \"text=New page\".\n"
        "- Only use CSS selectors if they are VERY simple and clearly visible in the provided HTML/text.\n"
        "- Do NOT invent deep/nth-child/nav-menu selectors.\n"
        "Return ONLY JSON with keys: action, url, selector, text, save_state, description, done."
        )
    )

    # system = SystemMessage(
    #     content=(
    #         "You are Agent B, planning how to operate arbitrary web apps via Agent A.\n"
    #         "Emit ONE atomic action as STRICT JSON.\n"
    #         "Allowed actions: 'goto', 'click', 'fill', 'press', 'wait_for', 'finish'.\n"
    #         "Return ONLY JSON with keys: action, url, selector, text, save_state, description, done.\n"
    #         "No extra keys, no markdown."
    #     )
    # )

    user = HumanMessage(
        content=(
            f"Task: {task}\n\n"
            f"Recent steps (truncated): {json.dumps(steps[-4:], indent=2)}\n\n"
            f"Current UI:\n{ui_context}\n\n"
            "Decide the next best atomic action."
        )
    )

    res = llm.invoke([system, user])
    print("\n[planner] raw LLM output:")
    print(res.content)

    # ---- 2a. Parse JSON (defensively) ----
    try:
        parsed = json.loads(res.content)
    except Exception:
        print("[planner] Invalid JSON from LLM, forcing finish.")
        parsed = {
            "action": "finish",
            "url": "",
            "selector": "",
            "text": "",
            "save_state": True,
            "description": "Invalid JSON from LLM.",
            "done": True,
        }

    # If it's a list, take first; else use as dict
    if isinstance(parsed, list) and parsed:
        action = parsed[0]
    else:
        action = parsed

    if not isinstance(action, dict):
        print("[planner] Non-dict action, forcing finish.")
        action = {
            "action": "finish",
            "url": "",
            "selector": "",
            "text": "",
            "save_state": True,
            "description": "Non-dict action from LLM.",
            "done": True,
        }

    # Ensure keys exist
    action.setdefault("url", "")
    action.setdefault("selector", "")
    action.setdefault("text", "")
    action.setdefault("save_state", False)
    action.setdefault("description", "")
    action.setdefault("done", False)
        # ---- 2b. Normalize weird shapes like:
    # { "action": { "action": "click", "selector": "#login-button", ... }, ... }
    raw_action = action.get("action")

    if isinstance(raw_action, dict):
        inner = raw_action

        # 1) Determine the actual action type
        inner_type = inner.get("action") or inner.get("type") or inner.get("name") or "finish"

        # 2) Hoist common fields from inner if missing on outer
        for key in ("selector", "url", "text"):
            if inner.get(key) and not action.get(key):
                action[key] = inner[key]

        for key in ("save_state", "description", "done"):
            # Use inner flags if not already set meaningfully outside
            if key in inner and (not action.get(key) and action.get(key) is not False):
                action[key] = inner[key]

        raw_action = inner_type  # now a primitive

    # If still not a string, coerce to string
    if not isinstance(raw_action, str):
        raw_action = str(raw_action)

    action["action"] = raw_action.strip().lower()


    # # ---- 2b. Normalize weird shapes like:
    # # { "action": { "action": "click", "selector": "#login-button" }, ... }
    # raw_action = action.get("action")

    # if isinstance(raw_action, dict):
    #     # Flatten nested dict into top-level fields
    #     inner = raw_action
    #     # Prefer explicit 'action' field inside
    #     inner_type = inner.get("action") or inner.get("type") or inner.get("name") or "finish"
    #     # Pull selector / url / text up only if not already set
    #     if inner.get("selector") and not action.get("selector"):
    #         action["selector"] = inner["selector"]
    #     if inner.get("url") and not action.get("url"):
    #         action["url"] = inner["url"]
    #     if inner.get("text") and not action.get("text"):
    #         action["text"] = inner["text"]
    #     raw_action = inner_type

    # # If still not a string, coerce
    # if not isinstance(raw_action, str):
    #     raw_action = str(raw_action)

    # action["action"] = raw_action.strip().lower()

    # ---- 2c. Validate against allowed actions ----
    allowed = {"goto", "click", "fill", "press", "wait_for", "finish"}
    if action["action"] not in allowed:
        print(f"[planner] Unsupported or malformed action '{action['action']}', forcing finish.")
        action = {
            "action": "finish",
            "url": "",
            "selector": "",
            "text": "",
            "save_state": True,
            "description": "LLM returned unsupported or malformed action.",
            "done": True,
        }

    print("[planner] parsed action:", action)

    # ---- 2d. Commit to state ----
    state["next_action"] = action
    state.setdefault("history", []).append({"role": "planner", "raw": res.content})

    if action.get("done") or action.get("action") == "finish":
        state["done"] = True

    return state


# ========== Executor Node (Agent A) ==========

def executor_node(state: AgentState) -> AgentState:
    global BROWSER
    if BROWSER is None:
        raise RuntimeError("BROWSER is not initialized. Use run_task().")

    action = state.get("next_action") or {}
    kind = action.get("action", "")

    if not kind or kind == "finish":
        state["done"] = True
        return state

    print(f"[executor] executing: {action}")

    # Run the action
    try:
        if kind == "goto":
            obs = BROWSER.goto(action.get("url", ""))
        elif kind == "click":
            obs = BROWSER.click(action.get("selector", ""))
        elif kind == "fill":
            obs = BROWSER.fill(action.get("selector", ""), action.get("text", ""))
        elif kind == "press":
            obs = BROWSER.press(action.get("text", "Enter"))
        elif kind == "wait_for":
            obs = BROWSER.wait_for(action.get("selector", ""))
        else:
            obs = {
                "url": BROWSER.page.url if BROWSER.page else "",
                "label": f"unknown_action:{kind}",
                "screenshot_path": "",
                "text_snippet": "",
                "state_hash": None,
            }
    except Exception as e:
        print(f"[executor] error while executing {kind}: {e}")
        obs = {
            "url": BROWSER.page.url if BROWSER.page else "",
            "label": f"error:{kind}",
            "error": str(e),
            "screenshot_path": "",
            "text_snippet": "",
            "state_hash": None,
        }

    # Decide whether to store this step
    steps = state.get("steps", [])
    save_state = bool(action.get("save_state", False))
    last_hash = state.get("last_state_hash")
    h = obs.get("state_hash")
    is_new_state = bool(h and h != last_hash)

    if save_state or is_new_state:
        steps.append({
            "action": action,
            "observation": {k: v for k, v in obs.items()},
        })
        state["steps"] = steps
        state["last_state_hash"] = h or last_hash
        print("[executor] recorded step. total steps:", len(steps))
    else:
        print("[executor] state not recorded (no save_state and no new hash).")

    # Clear next_action for next planner turn
    state["next_action"] = {}

    return state


# ========== Graph Wiring ==========

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)

    graph.add_edge(START, "planner")

    def route_from_planner(state: AgentState) -> str:
        return END if state.get("done") else "executor"

    graph.add_conditional_edges(
        "planner",
        route_from_planner,
        {END: END, "executor": "executor"},
    )

    graph.add_edge("executor", "planner")

    return graph.compile()


# ========== Orchestrator ==========
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

    with BrowserEnv(headless=headless) as browser:
        BROWSER = browser
        final_state = app.invoke(initial_state)

    BROWSER = None
    return final_state

if __name__ == "__main__":
    print("Enter tasks for Agent B to execute (one per line).")
    print("Example:")
    print("  Show me how to open Linear.")
    print("  Show me how to open Notion.")
    print("  Show me how to start creating a new project in Linear.")
    print("Press ENTER on an empty line when you're done.\n")

    demo_tasks: List[str] = []
    while True:
        line = input("Task: ").strip()
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

        # Set headless=False if you want to watch the browser.
        result = run_task(t, headless=False, max_steps=8)

        steps = result.get("steps", [])
        print(f"\nCaptured {len(steps)} key steps for task: {t!r}")

        if not steps:
            print("⚠️ No steps captured. Check planner logs above for JSON/action issues.")
            continue

        for i, step in enumerate(steps, start=1):
            action = step["action"]
            obs = step["observation"]
            print(f"\nStep {i}: {action.get('description','(no description)')}")
            print("  Action:     ", action.get("action"))
            if action.get("selector"):
                print("  Selector:   ", action.get("selector"))
            if action.get("url"):
                print("  Target URL: ", action.get("url"))
            print("  Page URL:   ", obs.get("url", ""))
            print("  Screenshot: ", obs.get("screenshot_path", ""))
            print("  Label:      ", obs.get("label", ""))

    print("\nAll tasks finished. Check the 'screenshots/' folder for captured UI states.")

# if __name__ == "__main__":
#     demo_tasks = [" how to open linear"]

#     for t in demo_tasks:
#         print("\n==============================")
#         print("Task:", t)
#         result = run_task(t, headless=True, max_steps=5)

#         steps = result.get("steps", [])
#         print(f"\nCaptured {len(steps)} key steps.")
#         for i, step in enumerate(steps, start=1):
#             action = step["action"]
#             obs = step["observation"]
#             print(f"\nStep {i}: {action.get('description','')}")
#             print("  Action:", action.get("action"))
#             print("  URL:", obs.get("url", ""))
#             print("  Screenshot:", obs.get("screenshot_path", ""))
#             print("  Label:", obs.get("label", ""))

#         if not steps:
#             print("No steps captured. Check [planner] logs above for JSON or action issues.")