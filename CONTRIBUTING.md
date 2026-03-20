# Contributing Guide

This project follows a `main + short-lived feature branches` workflow.

## 1) Branch Strategy

- `main`: always releasable, do not develop directly on it.
- All development goes to topic branches and lands via Pull Request (PR).

Recommended branch naming:

- `feat/<area>-<short-desc>`
- `fix/<area>-<short-desc>`
- `refactor/<area>-<short-desc>`
- `docs/<area>-<short-desc>`
- `chore/<area>-<short-desc>`
- `hotfix/<area>-<short-desc>` (urgent production fix)

Examples:

- `feat/libero-async-eval-watcher`
- `refactor/common-runtime-utils`
- `fix/robotwin-openpi-encoding`

`<area>` can be one of:

- `launcher`
- `core`
- `common`
- `libero`
- `robotwin`
- `real`
- `infra`

## 2) Commit Convention

We use Conventional Commits:

`<type>(<scope>): <subject>`

Types:

- `feat`: new feature
- `fix`: bug fix
- `refactor`: code restructuring without behavior change
- `perf`: performance improvement
- `docs`: documentation only
- `test`: tests only
- `build`: build/dependency changes
- `ci`: CI/CD only
- `chore`: misc maintenance

Examples:

- `feat(libero): extract async learner into utils module`
- `fix(common): handle empty replay batch safely`
- `refactor(robotwin): split env factory from train loop`
- `docs(repo): add branching and PR standards`

Breaking changes:

- `feat(core)!: ...`
- Or add footer: `BREAKING CHANGE: ...`

## 3) Daily Workflow

1. Sync main:
   - `git switch main`
   - `git pull --ff-only`
2. Create branch:
   - `git switch -c feat/<area>-<short-desc>`
3. Develop and commit in small logical chunks.
4. Push and open PR:
   - `git push -u origin <branch>`
5. Keep branch updated:
   - Preferred (clean history): `git fetch origin && git rebase origin/main`
   - Alternative (shared branch): `git fetch origin && git merge origin/main`
6. Merge after review + checks pass.

## 4) Pull Request Rules

- Keep PRs focused (single purpose).
- Include:
  - what changed
  - why
  - risk/regression points
  - test evidence
- Prefer squash merge for cleaner `main` history.

## 5) Recommended PR Size

- Ideal: <= 500 changed lines.
- If larger, split by module/concern into multiple PRs.

## 6) Release Safety

Before merge, ensure:

- Local lint/tests pass for changed modules.
- No accidental path hacks or hardcoded local absolute paths.
- Backward compatibility and migration notes are included when needed.

## 7) Optional Local Git Settings

Use commit template:

- `git config commit.template .gitmessage.txt`

Use rebase on pull by default:

- `git config pull.rebase true`

