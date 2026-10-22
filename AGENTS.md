# Repository Development Guidelines

## General principles

- Reuse existing code whenever possible.
- Prefer small, focused changes over large refactors.
- Follow existing naming, formatting, and architectural conventions.
- Avoid introducing new dependencies unless necessary.
- Don't try to preserve backward compatibility.
- Don't try to add defensive coding.
- Aggressively check to see how much have already been implemented
- Minimalise new code
- When commiting, separate to small incremental commits

## Style

- Commit message style: Conventional Commits 1.0.0 https://www.conventionalcommits.org/en/v1.0.0/#specification
- Short simple docstrings explaining, citing.
- No-types google style docstrings


## AI-assisted development workflow

### Phase 1: Conceptual analysis

Before implementing a feature:

1. Identify the components needed to build it from scratch.
2. Identify dependencies and integration points.
3. Focus only on essential functionality, no optional improvements.

### Phase 2: Repository investigation

Investigate the repository thoroughly before proposing changes.

- Search existing implementations, utilities, base classes, and abstractions.
- Read relevant call sites, tests, documentation, and configuration.
- Examine Git history and related branches when useful.
- Do not assume functionality is missing merely because it has a different name.
- Look for opportunities to reuse or extend existing implementations.

### Phase 3: Report and approval

Before changing code:

1. Give a concise report of what is already implemented, partially implemented, or missing.
2. Reference relevant files and functions.
3. Propose specific implementation tasks.
4. Present tasks as selectable checkboxes, grouped hierarchically when appropriate.
5. Separate required work from optional improvements.
6. Wait for explicit approval of the selected tasks.

Do not modify repository files during this investigation phase.

### Phase 4: Implementation

After approval:

- Implement only the selected tasks and their necessary dependencies.
- Reuse existing code instead of duplicating functionality.
- Add or update relevant tests.
- Run appropriate validation.
- Report changed files, changed number of lines, test results, and remaining issues.

## Repository-specific instructions

Document relevant setup commands, test commands, architectural conventions, and known constraints here.
