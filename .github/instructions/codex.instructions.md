---
description: General instructions for AI agents working on this repository.
applyTo: "**"
---

# General Principles

- Prefer simple, readable solutions over clever or highly optimized implementations.
- Implement only the requested functionality.
- Avoid introducing unnecessary dependencies.
- Keep functions small and focused.

# Architecture

- Maintain clear separation of concerns.
- Do not mix recognition, backend, frontend, and evaluation logic.
- Prefer modular code over large files.

# Code Quality

- Use type hints where appropriate.
- Add concise docstrings to public functions.
- Handle errors gracefully.
- Avoid duplicated code.
- Avoid dead code.

# Execution Policy

- Never execute scripts, tests, or applications automatically.
- Never use Python execution tools unless explicitly requested.
- After implementing a task, explain what changed.
- Tell me exactly which command I should run to verify the implementation.
- Wait for my feedback before making additional changes.

# Communication

- If requirements are ambiguous, ask or state assumptions instead of guessing.
- Briefly explain important architectural decisions.
- Keep explanations concise unless I ask for more detail.

# Scope

- Implement one logical task at a time.
- Do not begin the next task until I ask.
- Prefer incremental, reviewable changes over large implementations.
