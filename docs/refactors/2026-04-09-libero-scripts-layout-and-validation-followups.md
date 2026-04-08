# 2026-04-09 LIBERO Scripts Layout And Validation Follow-ups

## Summary

This follow-up cleans up the `examples/libero` workflow entrypoints after the
earlier runtime/package refactor work.

Main outcomes:

- move LIBERO workflow scripts into grouped directories under
  `examples/libero/scripts/`
- remove legacy flat script entrypoints instead of keeping compatibility wrappers
- rename data/eval/train/service scripts to action-oriented names
- update shell tools and README usage examples to the new script paths
- move auto-generated LIBERO local config out of the repo tree into a user cache dir
- keep task-9 smoke/validation configs used during recent end-to-end checks
- reduce spurious agentlace request-timeout failures by using an integer timeout value

## Final Script Layout

```text
examples/libero/scripts/
  train/
    run_actor.py
    run_learner.py
    launch_async_train.py
  eval/
    evaluate_checkpoint.py
    process_eval_queue.py
  data/
    collect_online_prefill.py
    prepare_offline_demos.py
    compute_normalization_stats.py
  services/
    serve_env.py
```

## Rationale

- script names should describe actions, not implementation roles
- `examples/libero/scripts` should reflect user workflows, not internal module names
- legacy flat script names added noise and encouraged drift in shell tools / docs
- generated machine-local LIBERO config should not live inside the repository

## Validation

The updated entrypoints were checked with:

- `python -m py_compile`
- `git diff --check`
- `--help` smoke checks for all grouped `train/eval/data/services` Python entrypoints
- `--help` smoke checks for the main `examples/libero/tools/*.sh` wrappers

Additional earlier validation already exercised:

- offline data preparation
- online prefill collection
- async train smoke
- eval smoke

## Notes

- Some historical analysis docs / generated figures still mention old script names.
  They are kept as historical references and do not affect runtime behavior.
