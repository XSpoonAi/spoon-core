# Skill Performance Analysis (SpoonReactSkill)

## Summary
SpoonReactSkill slowness and off-topic responses were traced to persistent, auto-activated skills leaking across turns and accumulating skill tools in the tool list. This caused repeated intent matching latency and bloated tool context, which increased response time and made later prompts drift toward irrelevant skill instructions/tools.

## Root Causes
1) **Auto-activated skills were never cleaned up**
   - `SpoonReactSkill.run()` auto-activated skills but did not deactivate them after the request finished.
   - Skill instructions remained in the system prompt and tools remained available for later, unrelated turns.

2) **Skill tool injection was one-way**
   - `SkillEnabledMixin._inject_skill_tools()` only added tools; it never removed tools for deactivated skills.
   - Over time, `available_tools` grew and polluted tool selection with stale skill tools.

3) **Intent matching always ran when enabled**
   - `SkillManager.find_matching_skills()` always attempted LLM-based intent matching even when trigger matches already existed.
   - This added avoidable latency to every turn that contained a trigger hit.

## Fixes Implemented
1) **Per-turn auto-skill activation with cleanup**
   - Added `_run_with_auto_skills()` to `SkillEnabledMixin` to activate skills for the current request and auto-deactivate them in a `finally` block.
   - `SpoonReactSkill.run()` now delegates to `_run_with_auto_skills()`, guaranteeing cleanup even on errors.

2) **Synchronized skill tool injection/removal**
   - Added `_sync_skill_tools()` to track which tool names were injected by skills and remove stale ones when skills deactivate.
   - `activate_skill()`, `deactivate_skill()`, and `deactivate_all_skills()` now call `_sync_skill_tools()`.
   - `ToolManager.remove_tool()` now safely no-ops if the tool is missing.

3) **Intent matching fallback only**
   - `find_matching_skills()` now skips LLM intent analysis when trigger matches exist, using intent only when triggers are empty.

## Expected Impact
- **Latency reduction**: Most trigger hits avoid LLM intent analysis, reducing per-turn overhead.
- **Improved relevance**: Auto-activated skills no longer persist beyond the turn, preventing instruction/tool bleed into later prompts.
- **Stable tool list**: Skill tools are added and removed in lockstep with activation, preventing tool list bloat over time.

## Compatibility Notes
- Manual skill activation remains persistent until explicitly deactivated.
- Auto-activated skills are now ephemeral per request; this is a behavioral change but matches expected auto-trigger semantics.
- Tool removal is limited to tools that were injected by skills, avoiding interference with non-skill tools.
