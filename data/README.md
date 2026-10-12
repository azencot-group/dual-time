# data/ - the training data tree

Five files, the in-house corpora of the two training stages
(`hf download Zisling/dualTslm-data --repo-type dataset --local-dir data`);
everything here except this file is gitignored. The training configs point
here with repo-relative paths (`data/stageA/...`, `data/stageB/...`).

```
data/
  stageA/
    chatts_style_captions.jsonl    32,000 rows   caption  (ChatTS-style captions)
    structural_captions.json        7,701 rows   caption  (structural captions)
  stageB/
    tsexam_perception.json         20,640 rows   MCQ      (TSExam perception)
    numerical_corpus.json          11,000 rows   scalar   (numerical corpus)
    reasoning_corpus.json         100,000 rows   MCQ + rationales (reasoning corpus)
```

The other training sources stay in the Hugging Face cache (set `HF_HOME` to
relocate it):
CaTS-Bench captions (`mhfisher/CaTSBench`, Stage A) and the ChatTS `ift` /
`sft` instruction sets (`ChatTSRepo/ChatTS-Training-Dataset`, Stage B).

Benchmark data lives outside this tree, under `eval_data/` (gitignored):
`tsrbench/` (umd-zhou-lab/TSRBench) and `chatts_eval/`
(ChatTS alignment sets, Zenodo 14349206); see the evaluation section of the
README for the download scripts.
