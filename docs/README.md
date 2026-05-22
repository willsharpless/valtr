# GitHub Pages

This folder is a static GitHub Pages app.

When `src/valtr/` changes, rebuild the browser bundle before publishing:

```bash
python scripts/build_web_bundle.py
```

To publish it:

1. Push this repo to GitHub.
2. In `Settings -> Pages`, choose `Deploy from a branch`.
3. Select your main branch and the `/docs` folder.

No GitHub Action is required for the current setup.
