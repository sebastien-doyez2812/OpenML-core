# OpenML, Open source Package for Machine learning


## Why OpenML?
Today's computer vision ecosystem is heavily fragmented. If you want YOLO, Ultralytics makes it trivial. But the moment you need advanced semantic segmentation, hybrid transformer backbones, or efficient fine-tuning on modern architectures, you end up stitching together disjointed codebases, mismatched tensor shapes, and unmaintained repositories.

OpenML bridges this gap by bringing a clean, Hugging Face-inspired Developer Experience (DX) to advanced computer vision tasks—from classical models to SOTA transformers and parameter-efficient fine-tuning.
## Supported Architectures:

- Classic & Deep Segmentation: U-Net, Attention U-Net, UNet++, Mask R-CNN

- Advanced Transformers: Swin Transformer, TransUNet, MobileUNetR

- SOTA & Edge: Vision Transformers, SAM 2 adaptations via efficient Low-Rank Adaptation (LoRA)
## Key features:

Unified Object-Oriented API: Every model inherits from a clean ModelBase class supporting standard .train(), .eval(), and .help() workflows.

Native LoRA Wrappers: Fine-tune massive vision models and transformers on modest hardware (like a single RTX 3070) without blowing up your VRAM.

Hugging Face Hub Integration: Push and pull checkpoints effortlessly using native .safetensors formatting.

MLflow Tracking Out-of-the-Box: Keep tabs on your losses, metrics, and hyperparameter logs without writing boilerplate monitoring code.

Interactive Help System: Running model.help() directly opens up the exact architectural documentation page on the web.

## Contributions:

Contributions, feature requests, and bug reports are welcome! Feel free to open an Issue or submit a Pull Request on GitHub.

## Author

Sébastien Doyez
ML & AI Engineer,
Interested in Agentics, Computer Vision, Robotics & Deep Learning