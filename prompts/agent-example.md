---
name: agent_global
description: Delegate to a nested Everruns agent through the Lua agent global.
promptforge: 0
max_tool_iterations: 4
capabilities:
  - promptforge/agent
tools:
  agent_run: promptforge/agent/run
models:
  writer: {}
---

# Agent Global

```lua
models.default("writer")
```

## Main

```lua
local coroutines = agent.run("In one sentence: what is a Lua coroutine?", {
  instructions = "You are terse. One short sentence, no preamble.",
})
local closures = agent.run("In one sentence: what is a Lua closure?", {
  instructions = "You are terse. One short sentence, no preamble.",
})
return "coroutine: " .. tostring(coroutines) .. "\nclosure: " .. tostring(closures)
```
