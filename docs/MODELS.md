# Models and Auxiliary Resources

## Multimodal backbones

| Key | Public model |
|---|---|
| qwen25vl_7b | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| qwen35_9b | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |

```bash
python download_resources.py --resource qwen25vl_7b
python download_resources.py --resource qwen35_9b
```

Downloads use the pinned public commits in `configs/models.json`, including model,
tokenizer and processor files. The pins identify this public reproduction version;
they are not an unsupported claim about the original training download revision.
The script records its source revision under `work/downloads/`. Follow each model's
license/model card. A Hugging Face token, if needed by your network or account,
belongs in your local environment, never in this repository.

## EfficientAD teacher

The included local detector is based on [rximg/EfficientAD](https://github.com/rximg/EfficientAD),
an unofficial implementation, not a claim to be the official implementation.

```bash
python download_resources.py --resource teacher
```

The teacher is `ckptSmall/best_teacher.pth` at upstream commit
`61d06b83667d28e306c88a592869f9f7b3e1b4f0`, downloaded through GitHub's LFS media
endpoint, not the small pointer file. Expected SHA-256:

```text
6e5e1220841d1604ee2830d55fb792a0363f955b8dff710691ca3f99aa884b5d
```

The downloader checks this hash before use. Category-specific student/autoencoder
weights are produced by `local_detector.py train`; no private trained detector is
required. Imagenette 320px supplies the training penalty images.

## SAEC local branch

```bash
python download_resources.py --resource yolo
```

This downloads `yolo11s-cls.pt` from the versioned Ultralytics asset release. See
[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) for the model and
license. The preserved comparison uses its class-confidence statistics as the
SAEC local branch; it is not a newly trained defect classifier. No additional
binary Qwen adapter is required because both routing methods use the same trained
three-field adapter in this controlled comparison.

The public file was downloaded and its SHA-256 is checked by the downloader:

```text
e2b605d1c8c212b434a75a32759a6f7adf1d2b29c35f76bdccd4c794cb653cf2
```

If HTTPS downloads fail, check your network/proxy configuration. Standard
`HTTPS_PROXY` settings can be used locally; never commit proxy credentials or
disable TLS verification to work around a failed connection.

## Project-specific adapters

There is no public pretrained UACR adapter bundle provided here. Run the training
commands to create `work/runs/<run>/adapters/<model>/<dataset>/`. For external
compatible adapters, pass `--adapter-root` explicitly and record their provenance.
Never substitute a different adapter silently or select it by test accuracy.
