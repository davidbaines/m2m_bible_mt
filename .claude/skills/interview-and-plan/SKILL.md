---
name: interview-and-plan
description: >-
  Turn a project brief into an agreed spec before any implementation: interview
  the user in depth, then write spec.md and todo.md and work against them. Reads
  a brief file (default: project-brief.md, or pass a path). Invoke with
  /interview-and-plan [brief-file].
argument-hint: "[brief-file]"
disable-model-invocation: true
---

# Interview → plan

You are turning a project brief into an agreed spec and task list before any
implementation begins.

## 1. Read the brief
Read the project brief first. Use the path in `$ARGUMENTS` if one was given,
otherwise `project-brief.md` in the working directory. It states the goal, the
constraints, and any design decisions already settled — treat those as fixed
unless we revisit them together.

## 2. Interview
Interview me in depth about every aspect of the plan until we reach a shared
understanding. Walk down each branch of the design tree, resolving dependencies
between decisions one at a time. Ask about requirements, edge cases, user
experience, data models, and failure modes. Fold each answer back into the
emerging design, and don't re-ask anything the brief already settles.

## 3. Write the files
Once we've reached shared understanding, and before starting implementation,
create two files:
1. `spec.md` — goals, implementation details, and a verification section
   describing exactly how you'll prove each piece works.
2. `todo.md` — a running task list you'll edit as you work, with complex tasks
   broken into verifiable sub-tasks.

## 4. Implementation phase
While working:
- Consult `spec.md` before every change.
- Mark each completed task in `todo.md` with `[x]` once it's done.
- Don't ask for clarification on anything you can resolve by reading the spec
  and running the tests. Start with the spec.
