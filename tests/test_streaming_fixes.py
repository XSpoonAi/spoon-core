"""
Tests for streaming infrastructure fixes:
1. ThreadSafeOutputQueue.put_nowait
2. BaseAgent.stream() while-loop condition
3. SpoonReactSkill.run() / SpoonReactAI.run() timeout parameter
"""

import asyncio
import inspect
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Test 1: ThreadSafeOutputQueue.put_nowait
# ---------------------------------------------------------------------------
async def test_output_queue_put_nowait():
    from spoon_ai.agents.base import ThreadSafeOutputQueue

    q = ThreadSafeOutputQueue()

    # put_nowait should exist and work
    assert hasattr(q, "put_nowait"), "put_nowait method missing"

    q.put_nowait({"content": "hello"})
    q.put_nowait({"tool_calls": []})
    assert q.qsize() == 2, f"Expected 2 items, got {q.qsize()}"

    item1 = await q.get(timeout=1.0)
    assert item1 == {"content": "hello"}, f"Wrong item: {item1}"

    item2 = await q.get(timeout=1.0)
    assert item2 == {"tool_calls": []}, f"Wrong item: {item2}"

    assert q.empty(), "Queue should be empty"

    print("  [PASS] ThreadSafeOutputQueue.put_nowait works correctly")


# ---------------------------------------------------------------------------
# Test 2: BaseAgent.stream() while-loop condition
# ---------------------------------------------------------------------------
async def test_stream_condition():
    """
    The old condition was:
        while not (task_done.is_set() or output_queue.empty()):
    which expands to:
        while (not task_done) and (not empty)
    This exits immediately when queue is empty (common at start).

    The fix changes it to:
        while not (task_done.is_set() and output_queue.empty()):
    which expands to:
        while (not task_done) or (not empty)
    This continues until task is done AND queue is drained.
    """
    from spoon_ai.agents.base import ThreadSafeOutputQueue

    # Simulate the fixed condition logic
    task_done = asyncio.Event()
    oq = ThreadSafeOutputQueue()

    # Initially: task not done, queue empty
    # Old: not (False or True) = not True = False → loop exits immediately (BUG)
    # New: not (False and True) = not False = True → loop continues (CORRECT)
    old_condition = not (task_done.is_set() or oq.empty())
    new_condition = not (task_done.is_set() and oq.empty())

    assert old_condition is False, "Old condition should exit (False) when queue is empty"
    assert new_condition is True, "New condition should continue (True) when task not done"

    # After items added: task not done, queue not empty
    oq.put_nowait("chunk1")
    old_after = not (task_done.is_set() or oq.empty())
    new_after = not (task_done.is_set() and oq.empty())
    assert old_after is True, "Old should continue when queue has items"
    assert new_after is True, "New should continue when queue has items"

    # Task done, queue has items → should still drain
    task_done.set()
    old_done_items = not (task_done.is_set() or oq.empty())
    new_done_items = not (task_done.is_set() and oq.empty())
    assert old_done_items is False, "Old exits even though queue has items (BUG)"
    assert new_done_items is True, "New continues to drain remaining items (CORRECT)"

    # Clean up
    await oq.get(timeout=1.0)

    # Task done, queue empty → both should exit
    old_final = not (task_done.is_set() or oq.empty())
    new_final = not (task_done.is_set() and oq.empty())
    assert old_final is False
    assert new_final is False  # Both exit when done AND empty

    print("  [PASS] stream() while-loop condition fixed correctly")


# ---------------------------------------------------------------------------
# Test 3: Verify the actual source code has the fix
# ---------------------------------------------------------------------------
def test_stream_source_code():
    from spoon_ai.agents.base import BaseAgent

    source = inspect.getsource(BaseAgent.stream)
    # The fix changes "or" to "and" in the while condition
    assert "self.task_done.is_set() and self.output_queue.empty()" in source, \
        "stream() source should use 'and' (not 'or') in while condition"

    # Should NOT have the old buggy condition
    # (it's OK if "or" appears elsewhere in the source, just not in this specific pattern)
    assert "while not (self.task_done.is_set() or self.output_queue.empty())" not in source, \
        "stream() should not have the old buggy 'or' condition"

    print("  [PASS] stream() source code has correct while-loop condition")


# ---------------------------------------------------------------------------
# Test 4: SpoonReactSkill.run() accepts timeout
# ---------------------------------------------------------------------------
def test_run_timeout_signature():
    from spoon_ai.agents.spoon_react_skill import SpoonReactSkill
    from spoon_ai.agents.spoon_react import SpoonReactAI

    # Check SpoonReactSkill.run signature
    sig_skill = inspect.signature(SpoonReactSkill.run)
    params_skill = list(sig_skill.parameters.keys())
    assert "timeout" in params_skill, \
        f"SpoonReactSkill.run() missing 'timeout' param. Has: {params_skill}"

    # Check SpoonReactAI.run signature
    sig_ai = inspect.signature(SpoonReactAI.run)
    params_ai = list(sig_ai.parameters.keys())
    assert "timeout" in params_ai, \
        f"SpoonReactAI.run() missing 'timeout' param. Has: {params_ai}"

    print("  [PASS] SpoonReactSkill.run() and SpoonReactAI.run() accept timeout")


# ---------------------------------------------------------------------------
# Test 5: Full streaming integration test
# ---------------------------------------------------------------------------
async def test_stream_integration():
    """Test that put_nowait + fixed stream loop works end-to-end."""
    from spoon_ai.agents.base import ThreadSafeOutputQueue

    task_done = asyncio.Event()
    oq = ThreadSafeOutputQueue()
    collected = []

    async def producer():
        """Simulate agent step() putting items in queue."""
        await asyncio.sleep(0.1)  # Simulate LLM delay
        oq.put_nowait({"content": "Hello"})
        oq.put_nowait({"tool_calls": [{"name": "shell"}]})
        await asyncio.sleep(0.1)  # Simulate tool execution
        oq.put_nowait({"content": " World"})
        task_done.set()

    async def consumer():
        """Simulate the fixed stream() loop logic."""
        while not (task_done.is_set() and oq.empty()):
            try:
                item = await asyncio.wait_for(oq.get(timeout=1.0), timeout=1.0)
                collected.append(item)
            except asyncio.TimeoutError:
                continue

    # Run producer and consumer concurrently
    await asyncio.gather(producer(), consumer())

    assert len(collected) == 3, f"Expected 3 chunks, got {len(collected)}: {collected}"
    assert collected[0] == {"content": "Hello"}
    assert collected[1] == {"tool_calls": [{"name": "shell"}]}
    assert collected[2] == {"content": " World"}

    print("  [PASS] Full streaming integration (put_nowait + fixed loop) works")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    print("=" * 60)
    print("Running spoon-core streaming infrastructure tests")
    print("=" * 60)

    passed = 0
    failed = 0

    tests = [
        ("Test 1: ThreadSafeOutputQueue.put_nowait", test_output_queue_put_nowait),
        ("Test 2: stream() condition logic", test_stream_condition),
        ("Test 3: stream() source code check", test_stream_source_code),
        ("Test 4: run() timeout signature", test_run_timeout_signature),
        ("Test 5: Stream integration", test_stream_integration),
    ]

    for name, test_fn in tests:
        try:
            print(f"\n{name}:")
            if asyncio.iscoroutinefunction(test_fn):
                await test_fn()
            else:
                test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
