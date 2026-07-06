# Note for the ClearML worker administrators

Subject: ClearML agent fails to start tasks on the aqua-gpu-dallas workers
(pkg_resources / setuptools)

## Summary

Tasks sent to the `jobs_backlog` queue fail during the ClearML agent's own
start-up inside the task container, before any user code runs. The worker pulls
the task and starts the container correctly, so queueing and scheduling are
fine; the failure is in the agent bootstrap.

## What happens

The worker runs each task in a fresh `python:3.12-bullseye` container and, inside
it, installs `clearml-agent==2.0.4` and then runs `python -m clearml_agent
execute`. That import chain fails:

```
File ".../clearml_agent/external/requirements_parser/requirement.py", line 3
    from pkg_resources import Requirement as Req
ModuleNotFoundError: No module named 'pkg_resources'
Process failed, exit code 1
```

## Cause

The container installs a recent `setuptools` (version 83 was pulled in) as a
dependency of clearml-agent. Recent setuptools no longer provides the
`pkg_resources` module, which `clearml-agent 2.0.4` still imports. The two are
therefore incompatible in a clean Python 3.12 environment.

## Suggested fixes (any one is sufficient)

1. Pin `setuptools<81` in the worker or container environment, for example via
   the agent's extra pip install, or in the default docker image.
2. Upgrade `clearml-agent` to a release that does not import `pkg_resources`.
3. Set the agents' default docker image to one that already has a compatible
   setuptools and clearml-agent (for example a CUDA or PyTorch image that your
   working jobs already use), rather than the bare `python:3.12-bullseye`.

## Reference

- Queue: `jobs_backlog`
- Worker: `aqua-gpu-dallas:gpu7` (all `aqua-gpu-dallas:gpu0-7` are affected)
- Example failed task id: `d979edaa28fd418a887e92b0bbf01db3`
- Server: `app.sil.hosted.allegro.ai`

Tasks that specify their own docker image may be unaffected, so this mainly hits
jobs that rely on the default image.
