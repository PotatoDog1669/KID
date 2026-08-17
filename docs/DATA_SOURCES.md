# Data Sources

Datasets are intentionally excluded from this repository. Obtain each dataset directly from its original release, comply with its license and access terms, and retain the source version in your local experiment record.

| Paper task | Original dataset |
| --- | --- |
| Hateful Memes | Meta AI Hateful Memes dataset |
| HarMeme | HarMeme dataset release from the original authors |
| MAMI Task A/B | Multimedia Automatic Misogyny Identification shared task release |
| ToxiCN-MM Task A/B | ToxiCN-MM dataset release from the original authors |
| BanglaAbuseMeme Task A/B | BanglaAbuseMeme dataset release from the original authors |

For each download, record the official landing page or repository URL, release date, version or checksum, and license in a local `data_manifest.md` that remains untracked. After processing, keep images and annotations outside Git and only expose documented preprocessing commands and dataset schemas.

The KID repository expects locally registered dataset names through `TRAIN_DATASET` and `EVAL_DATASET`; it deliberately does not provide a prepopulated registry with machine-specific paths.
