# Contributing

Open an issue before large architectural changes. Every change should include the affected data contract or protocol update and a focused test.

Before opening a pull request:

```bash
python -m compileall cc_hhgt scripts runtime website/backend
pytest -q tests
```

Do not include generated datasets, checkpoints, private paths, credentials, or local environment directories in commits. Keep model and website outputs reproducible from a declared config and input manifest.
