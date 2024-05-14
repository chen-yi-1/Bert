
Build via:

```
docker build -t v5 .
```

Run via:

```
tira-run \
  --input-dataset generative-ai-authorship-verification-panclef-2024/pan24-generative-authorship-tiny-smoke-20240417-training \
  --image v5 \
  --tira-vm-id team-chenteam-chen123hhh \
  --skip-local-test \
  --push true
```

