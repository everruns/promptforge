---
name: agent_example
description: Delegate sub-tasks to a nested Everruns agent through the Lua agent2 global.
promptforge: 0
max_tool_iterations: 4
capabilities:
  - promptforge/agent
tools:
  agent: promptforge/agent/run
models:
  writer: {}
---

# Agent Example

```lua
models.default("writer")
```

## Main

```lua
local coroutines = agent2.run("In one sentence: what is a Lua coroutine?", {
  instructions = "You are terse. One short sentence, no preamble.",
})
local closures = agent2.run("In one sentence: what is a Lua closure?", {
  instructions = "You are terse. One short sentence, no preamble.",
})
return "coroutine: " .. tostring(coroutines) .. "\nclosure: " .. tostring(closures)
```
