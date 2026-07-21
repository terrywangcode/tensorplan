# Contributing

Issues and pull requests are welcome, especially for incorrect model architecture fields, new GPU specifications, and reproducible throughput data.

## Local development

```bash
python3 -m http.server 4173
```

Open `http://localhost:4173`, then run the checks before submitting a change:

```bash
node model-config.test.js
node static-site.test.js
```

When adding a model or GPU, cite the vendor or model repository source in the pull request. Planning factors should remain conservative and clearly labeled as estimates.
