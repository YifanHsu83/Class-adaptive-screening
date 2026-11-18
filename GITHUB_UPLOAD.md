# Updating the GitHub repository

Project repository: https://github.com/YifanHsu83/Class-adaptive-screening

Clone the repository, create an environment, and run the checks before editing:

```powershell
git clone https://github.com/YifanHsu83/Class-adaptive-screening.git
cd Class-adaptive-screening
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
git status
git diff
```

After making a change, review and publish only the files you intended to edit:

```powershell
git status
git diff
git add <changed-files>
git diff --cached
git commit -m "Describe your changes"
git push
```

Replace `<changed-files>` with explicit paths, for example
`src/class_adaptive_screening/cutoffs.py README.md`. Avoid `git add .` when the
repository is inside a larger research directory. If Git asks for an identity,
set your own name and GitHub email with `git config user.name` and
`git config user.email`. Never place a password or access token in the repository
or remote URL.

## Verify the upload

Open the repository page and check the README, source files, and the Actions
test run. Raw data, generated results, local dataset paths, and Python caches
should not appear. Copy the repository URL into the paper's code-availability
statement after the upload is complete.

The GitHub Actions workflow runs Ruff and pytest after every push and pull
request. A green check beside the commit confirms that both checks passed.
