# Uploading the code to GitHub

Project repository: https://github.com/YifanHsu83/Class-adaptive-screening

## Updating this existing repository

Clone the repository, edit your files, run the tests, then commit and push:

```powershell
git clone https://github.com/YifanHsu83/Class-adaptive-screening.git
cd Class-adaptive-screening
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
git status
git diff
git add src tests examples README.md
git diff --cached
git commit -m "Describe your changes"
git push
```

Adjust the `git add` list to the files you actually changed. If you already have
a clone, use it instead of cloning again. The remaining instructions explain
the initial setup for a new, empty repository.

Run the commands below from the `class-adaptive-screening` folder, not the parent
research folder. This keeps unrelated research files out of the repository.

## 1. Check the code locally

Follow the installation and synthetic demo commands in `README.md`. The main
experiments require prepared feature files; the demo does not.

## 2. Create an empty GitHub repository

Sign in to GitHub and create a repository named `class-adaptive-screening`, or
another name you prefer. Choose Public if the code should be accessible to
readers, or Private while preparing it. Leave the README, license, and gitignore
initialization options unchecked: these files already exist locally.

Keep the existing author and license metadata only if it is correct for this
project. `CITATION.cff` supplies GitHub's citation information.

## 3. Commit and push

Replace `YOUR-USERNAME` with the repository owner. These commands assume a new,
empty remote repository. For an existing remote with commits, inspect its history
before pushing; do not force-push over it.

```powershell
git init
git add .github .gitignore CITATION.cff LICENSE README.md GITHUB_UPLOAD.md pyproject.toml configs/experiment.json configs/datasets.example.json data/README.md src tests examples
git diff --cached --stat
git diff --cached
git commit -m "Add class-adaptive screening code and reproducibility instructions"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/class-adaptive-screening.git
git push -u origin main
```

If Git requests an author identity, configure your own name and email locally
with `git config user.name "Your Name"` and
`git config user.email "your GitHub email"`, then repeat the commit. Use GitHub's
sign-in prompt or a credential manager when pushing; do not put passwords or
tokens in source files or remote URLs.

If `origin` already exists, inspect `git remote -v` rather than adding it again.
If Git reports dubious ownership, verify that this is your project, then add only
this repository's absolute path with `git config --global --add safe.directory`.

## 4. Verify the upload

Open the repository page and check the README, source files, and the Actions
test run. Raw data, generated results, local dataset paths, and Python caches
should not appear. Copy the repository URL into the paper's code-availability
statement after the upload is complete.

## Later updates

Review changes with `git status` and `git diff`, stage the intended files with
`git add`, commit with a short description, and run `git push`.
