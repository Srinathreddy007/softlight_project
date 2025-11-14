# Softlight UI State Capture Agent

This repo is my solution for the **Softlight Engineering Take-Home**.  
It builds a small multi-agent system that can follow natural-language instructions
(e.g. *“Create a project in Linear”*, *“Filter a database in Notion”*) and
automatically:

- Navigate the live web app in a real browser
- Execute each UI step (clicks, fills, key presses, etc.)
- Capture screenshots + metadata for every important UI state

The output is a per-task **dataset of UI states** that could be used to build
tutorials, demos, or training data.

## Requirements:
  - `pip install -r requirements.txt`
  - `playwright install`

## How it Works

### Two agents (LangGraph)

- **Planner (Agent B – `planner_node`)**
  - Uses `gpt-4o-mini` via `langchain-openai`.
  - Sees the task, recent history, page text, a list of “affordances”
    (labels of visible buttons/links), and the latest screenshot.
  - Returns **one atomic action** as strict JSON:

    ```json
    {
      "action": "goto | click | fill | press | wait_for | hover | upload | finish",
      "target": { "role": "button", "name": "New page" },
      "value": "...",
      "url": "...",
      "save_state": true,
      "description": "...",
      "done": false
    }
    ```

  - On the first step, it chooses a starting URL

- **Executor (Agent A – `executor_node` + `BrowserEnv`)**
  - Runs in a **persistent Playwright Chromium context** (`playwright_data/`)
    so logins survive across tasks.
  - Executes the planner’s action using flexible helpers:
    - `goto(url)`
    - `click_flexible(Target)`
    - `fill_flexible(Target, text)` (rich-text / contenteditable aware)
    - `press_keys("Control+K, ... , Enter")`
    - `wait_for_flexible`, `hover`, `upload`
  - After each step it captures an **observation** with:
    - Full-page **screenshot**
    - **URL**
    - **Text snippet** from `<body>`
    - Clickable **affordances** (labels from buttons, links, menu items)
    - A **state hash** (URL + text) to detect UI changes

LangGraph connects them in a loop:


START → planner → executor → planner → ... → END

<img width="703" height="707" alt="image" src="https://github.com/user-attachments/assets/35fa553b-5624-4471-aaeb-12554fef26f1" />
