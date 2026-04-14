# MLEvolve Architecture

## Overview

MLEvolve uses **Monte Carlo Graph Search (MCGS)** to explore a tree of ML pipeline solutions. An LLM generates code, a subprocess executes it, another LLM parses the results, and backpropagation updates the tree — guiding the next iteration toward better solutions.

```
run.py                          Entry point, thread pool orchestration
  |
  v
engine/agent_search.py          Brain: node selection -> action -> execute -> evaluate
  |
  +-- engine/node_selection.py   UCT tree descent or Top-K selection
  +-- agents/*_agent.py          Code generation (draft, improve, debug, fusion, evolution)
  +-- agents/code_review_agent   LLM-based code review before execution
  +-- engine/executor.py         Subprocess execution with isolation
  +-- agents/result_parse_agent  LLM extracts metric from stdout
  +-- engine/evaluation.py       Reward computation and backpropagation
  +-- engine/solution_manager.py Top-K tracking and best solution persistence
```

## Main Loop (`run.py`)

Two phases:

1. **Phase 1 — Sequential Drafts**: Generate `initial_drafts` independent solutions (code only, no execution yet). Each becomes a separate branch in the search tree.

2. **Phase 2 — Parallel Search**: A `ThreadPoolExecutor` runs pending draft executions and new search steps concurrently. Each completed step feeds back into the next iteration.

```
Phase 1: [draft_1] -> [draft_2]           (code generation only)
Phase 2: [exec draft_1] [exec draft_2]    (execute + search in parallel)
          [step_3] [step_4] ...
```

## Single Step Call Chain

When `agent.step()` is called:

```
1. SELECT NODE
   node_selection.select_with_soft_switch()
   +-- Early (< 50% time):  UCT descent (exploration)
   +-- Mid   (50-70%):      Weighted mix of UCT + Top-K
   +-- Late  (> 70%):       Top-K exploitation

2. CHOOSE ACTION (based on selected node's state)
   +-- Root node           -> draft_agent        (new solution)
   +-- Buggy parent        -> debug_agent        (fix the bug)
   +-- Good parent:
       +-- Stagnant branch -> evolution_agent     (learn from trajectory)
       +-- Stagnant + late -> fusion_agent        (cross-branch merge)
       +-- Otherwise       -> improve_agent       (incremental improvement)

3. CODE REVIEW
   code_review_agent.run()
   +-- LLM checks for data leakage, logic errors
   +-- Applies SEARCH/REPLACE patches if needed

4. EXECUTE
   executor.run(code, node_id)
   +-- Writes code to runfile_{slot}.py
   +-- Runs in subprocess with timeout
   +-- Captures stdout/stderr and exceptions
   +-- Returns ExecutionResult

5. PARSE RESULTS
   result_parse_agent.run(node, exec_result)
   +-- LLM extracts: is_bug, metric, summary, lower_is_better
   +-- Validates metric direction matches global setting
   +-- Sets node.is_buggy, node.metric, node.analysis

6. VALIDATE
   execution.validate_executed_node()
   +-- Flags metric=0.0 as buggy (when maximizing)
   +-- Registers successful nodes to branch tracker

7. EVALUATE & BACKPROPAGATE
   evaluation.check_improvement(node, parent)
   +-- Improvement > threshold?  -> continue improving (no backprop)
   +-- Stagnated too long?       -> backpropagate reward up tree
   +-- Reward: +1.5 (beat best), +1 (improved), 0 (neutral), -1 (buggy)

8. UPDATE BEST
   solution_manager.update_best_solution()
   +-- Maintain top-K candidates (branch-diverse)
   +-- Save best_solution/solution.py to disk
```

## Search Tree

```
         [virtual_root]
        /       |       \
   [draft_1] [draft_2] [draft_3]       <- branches
      |          |
   [improve]  [improve]
      |          |
   [improve]  [debug]                  <- fix buggy node
      |          |
   [evolution] [improve]               <- learn from trajectory
      |
   [fusion]                            <- merge ideas from draft_2
```

Each node stores:
- `code` — the Python solution
- `metric` — validation score (with maximize/minimize direction)
- `is_buggy` — whether execution failed
- `visits`, `total_reward` — MCTS statistics for UCT
- `branch_id` — which branch this belongs to
- `stage` — draft / improve / debug / fusion / evolution

## Node Selection: UCT vs Top-K

**UCT (exploration)** — standard MCTS formula:
```
UCT = (total_reward / visits) + C * sqrt(ln(parent_visits) / visits)
```
Unvisited nodes get infinity (highest priority). Descends tree picking best UCT child.

**Top-K (exploitation)** — ranks all successful nodes globally by metric, weighted random selection (rank 1 gets 4x weight of rank 4). Enforces branch diversity (max N per branch).

The system transitions from UCT to Top-K as time progresses (controlled by `explore_switch_start/end`).

## Code Generation Agents

| Agent | When | What it does |
|-------|------|--------------|
| **draft** | Root node, under draft limit | Creates a new independent solution from scratch (stepwise: data->model->training) |
| **improve** | Good parent, not stagnant | Incremental improvement via SEARCH/REPLACE diff patches |
| **debug** | Buggy parent | Reads error trace, applies targeted fix |
| **evolution** | Branch stagnant | Analyzes full branch trajectory (history of changes), proposes informed improvement |
| **fusion** | Branch stagnant + late stage | Merges techniques from a different branch's best solution into current one |
| **aggregation** | Root node, draft limit reached | Ensemble combining best ideas from multiple branches |

All agents share:
- Task description + data preview as context
- `impl_guideline` with execution constraints
- Memory of sibling attempts (what worked / failed at this tree position)

## Stepwise Code Generation (`agents/coder/stepwise_coder.py`)

Draft solutions are built in 3 specialized stages, then merged:

```
Step 1: Data Processing & Feature Engineering
         (load data, clean, encode, split)
              |
Step 2: Model Design
         (architecture, loss function, optimizer)
              |
Step 3: Training & Evaluation
         (training loop, validation metric, output)
              |
MetaAgent: Merge all steps into single runnable script
```

## Key Files

| Component | Path |
|-----------|------|
| Entry point | `run.py` |
| Search orchestrator | `engine/agent_search.py` |
| Node selection | `engine/node_selection.py` |
| Tree node | `engine/search_node.py` |
| Code executor | `engine/executor.py` |
| Result parser | `agents/result_parse_agent.py` |
| Evaluation | `engine/evaluation.py` |
| Solution tracking | `engine/solution_manager.py` |
| Draft agent | `agents/draft_agent.py` |
| Improve agent | `agents/improve_agent.py` |
| Debug agent | `agents/debug_agent.py` |
| Fusion agent | `agents/fusion_agent.py` |
| Evolution agent | `agents/evolution_agent.py` |
| Aggregation agent | `agents/aggregation_agent.py` |
| Code review | `agents/code_review_agent.py` |
| Stepwise coder | `agents/coder/stepwise_coder.py` |
| Prompt templates | `agents/prompts/` |
| LLM interface | `llm/__init__.py`, `llm/gemini.py` |
| Conditions/triggers | `engine/conditions.py`, `agents/triggers.py` |
| Config | `config/__init__.py`, `config/config.yaml` |

## Config Quick Reference

```yaml
agent:
  steps: 10               # total search iterations
  time_limit: 7200         # wall-clock limit (seconds)
  initial_drafts: 2        # independent starting solutions

  search:
    parallel_search_num: 1           # concurrent execution slots
    num_drafts: 5                    # max drafts from root
    num_improves: 3                  # max improvements per node
    metric_improvement_threshold: 0.0001
    max_improve_failure: 3           # stagnation patience
    branch_stagnation_threshold: 3   # triggers fusion/evolution

  decay:
    exploration_constant: 1.414      # UCT exploration weight (C)
```
