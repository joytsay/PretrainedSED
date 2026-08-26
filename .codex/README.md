# Codex Instructions

## Runtime Context

This repository is normally used inside Docker.

Path mappings:

- `/home/joy/Git/PretrainedSED` on the host maps to `/workspace` in the container.
- `/mnt/data` on the host maps to `/data` in the container.

The container is started with:

```bash
docker run --rm --gpus all -it \
  --name psed \
  --shm-size=16g \
  -p 7862:7862 \
  -v "/home/joy/Git/PretrainedSED:/workspace" \
  -v "/mnt/data:/data" \
  psed-snapshot:latest \
  /bin/bash
```

## Working Rules

- Treat `/workspace` as the canonical repository path when referring to files.
- Do not run Python scripts automatically.
- Do not run training, evaluation, inference, or other project code unless explicitly asked.
- After modifying files, do not run validation, syntax checks, git diff/status checks, or other post-edit checks unless explicitly asked. Do not reply "Per your repo rule, I did not run syntax checks, git checks, or validation after editing." or "I did not run post-edit checks."outputs logs.
- Do not install packages. Create or edit code only; the user will run dependency installation and validation manually.
- Keep source edits focused and summarize the changes after editing.
- This is machine learning training code. Preserve experiment behavior unless a behavior change is clearly requested.
- When creating a Python file, include a short summary comment on the top of the file in the response that explains why it was created and what the functions or scripts do.
