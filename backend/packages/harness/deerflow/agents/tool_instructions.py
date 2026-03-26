"""Shared tool usage instructions injected into all agent system prompts.

This ensures every agent (lead, general-purpose subagent, bash subagent)
uses the correct parameter names and ordering for sandbox tools.
"""

TOOL_USAGE_GUIDE = """<tool_usage_guide>
**CRITICAL: Every sandbox tool requires a `description` parameter as the FIRST argument. You MUST include it.**
**NEVER use `bash` with echo/cat/heredoc to write files — ALWAYS use the `write_file` tool instead.**

**write_file** — Write text content to a file:
```
write_file(
    description="saving hoodie ground truth data",
    path="/mnt/user-data/workspace/docs/ground-truth/hoodie.json",
    content="<file contents here>"
)
```
Required parameters (in order): `description`, `path`, `content`

**bash** — Execute a bash command:
```
bash(
    description="listing workspace files",
    command="ls -la /mnt/user-data/workspace"
)
```
Required parameters (in order): `description`, `command`

**read_file** — Read a text file:
```
read_file(
    description="reading config file",
    path="/mnt/user-data/workspace/config.json"
)
```
Required parameters (in order): `description`, `path`

**str_replace** — Replace text in a file:
```
str_replace(
    description="fixing typo in readme",
    path="/mnt/user-data/workspace/README.md",
    old_str="helo world",
    new_str="hello world"
)
```
Required parameters (in order): `description`, `path`, `old_str`, `new_str`

**ls** — List directory contents:
```
ls(
    description="checking project structure",
    path="/mnt/user-data/workspace"
)
```
Required parameters (in order): `description`, `path`

**COMMON MISTAKES TO AVOID:**
- ❌ DO NOT use `filename` — the parameter is called `path`
- ❌ DO NOT omit `description` — it is REQUIRED as the first parameter on every tool
- ❌ DO NOT use relative paths — always use absolute paths starting with `/mnt/user-data/`
- ❌ DO NOT use `bash` with echo/cat/heredoc to create files — use `write_file` instead
- ✅ ALWAYS provide `description` first, then the other parameters in order
</tool_usage_guide>"""
