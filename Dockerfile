# docker build -t v5 .
FROM registry.webis.de/code-research/tira/tira-user-team-chenteam-chen123hhh/tira:v4-tira-docker-software-id-resonnt-halite

ADD test.py /workspace/clef/test.py

ENTRYPOINT [ "python", "/workspace/clef/test.py", "$inputDataset/dataset.jsonl", "$outputDir" ]

