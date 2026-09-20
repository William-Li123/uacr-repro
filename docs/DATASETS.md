# Public Dataset Preparation

Download each dataset from its owner, comply with its license and retain the
original filenames/case. Do not download a re-split mirror: the project uses the
fixed manifests in `data/three_field_qwen/`, not each provider's evaluation split
as-is. A subset of original test anomalies is used for supervised training; report
this protocol explicitly rather than claiming the standard unsupervised benchmark.

| Dataset | Official source | Directory under `work/raw/` |
|---|---|---|
| MVTec AD (original, not AD2) | [MVTec](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) | `mvtec_ad/` |
| MVTec LOCO AD | [MVTec](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad) | `mvtec_loco/` |
| VisA | [Amazon SPot-the-Difference](https://github.com/amazon-science/spot-diff), [AWS registry](https://registry.opendata.aws/visa/) | `visa/` |
| KolektorSDD2 | [ViCoS Lab](https://www.vicos.si/resources/kolektorsdd2/) | `ksdd2/` |
| GoodsAD | [Dataset authors](https://github.com/jianzhang96/GoodsAD) | `goodsad/` |
| Imagenette 320px | [fastai](https://github.com/fastai/imagenette) | `imagenette2-320/` |

The MVTec pages use a download form. KSDD2 provides a download link on its page;
GoodsAD provides category downloads. Obtain all categories when evaluating an
entire dataset. This project supplies no substitute credentials or unofficial
redistributed image bundles. The datasets retain their upstream licenses, including
non-commercial restrictions where applicable; this repository does not relicense them.

## Expected extraction layout

Remove any extra top-level wrapper directory so the following paths exist:

```text
work/raw/
  mvtec_ad/bottle/train/good/000.png
  mvtec_ad/bottle/test/broken_large/000.png
  mvtec_loco/breakfast_box/train/good/000.png
  mvtec_loco/breakfast_box/validation/good/000.png
  mvtec_loco/breakfast_box/test/logical_anomalies/000.png
  visa/candle/Data/Images/Normal/0000.JPG
  visa/candle/Data/Images/Anomaly/001.JPG
  ksdd2/train/10000.png
  ksdd2/test/<original-image-name>.png
  goodsad/cigarette_box/train/good/000_000.jpg
  imagenette2-320/train/<class>/<image>.JPEG
```

These examples illustrate layout, not a requirement that every example filename
is present in every dataset release. The manifests specify exact files. VisA
paths refer to the original JPGs, not renamed PNG aliases. LOCO official validation
normals are referenced by their original `validation/good` paths. KSDD2 masks are
not inputs to the multimodal model; the provided label/split manifests are fixed.

```bash
python download_resources.py --resource visa --extract
python download_resources.py --resource imagenette --extract
python bootstrap.py --raw-root work/raw --check-only
python bootstrap.py --raw-root work/raw
```

For another dataset location, pass its parent directory to `--raw-root`. It must
contain the named dataset subdirectories above. `--check-only` fails on the first
missing file; it never silently drops samples or changes split membership.

## Fixed split counts

| Manifest key | Train | Validation | Test |
|---|---:|---:|---:|
| goodsad_80p | 3337 | 1114 | 1673 |
| mvtec_ad_80p | 3532 | 732 | 1090 |
| mvtec_loco_80p | 2162 | 594 | 895 |
| visa_80p | 7506 | 1299 | 2016 |
| ksdd2_mvtlike | 2087 | 402 | 602 |

KSDD2's training manifest contains normal images only and is used for the local
detector, not a dedicated three-field adapter. Counts are those of this project,
not a claim to equal a provider's official train/test counts.

The committed prompts and text targets are versioned experiment annotations
(including templates and coarse labels), not newly acquired human annotations.
Their `source.text_source` fields identify origins. Preserve them to reproduce the
same heuristic metrics; do not interpret ExplainHit as a human-rated explanation score.
