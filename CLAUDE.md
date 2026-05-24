# Repository conventions

## Git workflow — `main` is protected

`main` requires all changes to go through a **pull request**. Direct
`git push` to `main` is rejected by a branch-protection rule
(`GH013: Changes must be made through a pull request`).

**Do not** suggest or perform `git checkout main && git merge … && git push`.
Instead, land work via a PR:

```bash
# from the feature branch, with work committed:
git push -u origin <feature-branch>
gh pr create --base main --head <feature-branch> --fill
# then merge the PR on GitHub (or: gh pr merge --squash/--merge)
```

If a local merge into `main` was already made but can't be pushed, recover with:

```bash
git checkout main
git reset --hard origin/main      # discard the unpushable local merge
git checkout <feature-branch>      # the commits still live here
```

## Shell

The user's interactive **zsh does not treat `#` as a comment** (no
`interactive_comments`). Do **not** put trailing `# …` comments on shell
commands you hand the user to paste — they get parsed as arguments
(e.g. `git push  # publish` → `git push '#' publish …`). Keep handed-over
commands comment-free, or put explanations on separate lines / in prose.
