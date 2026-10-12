# OpenML-core, Open source Package for Machine learning


## Why OpenML?
Today's computer vision ecosystem is heavily fragmented. If you want YOLO, Ultralytics makes it trivial. But the moment you need advanced semantic segmentation, hybrid transformer backbones, or efficient fine-tuning on modern architectures, you end up stitching together disjointed codebases, mismatched tensor shapes, and unmaintained repositories.

OpenML bridges this gap by bringing a clean, Hugging Face-inspired Developer Experience (DX) to advanced computer vision tasks—from classical models to SOTA transformers and parameter-efficient fine-tuning.
## Supported Architectures:

- Classic & Deep Segmentation: U-Net, Attention U-Net, UNet++

- Advanced Transformers: MobileUnetR, SegFormer

- SOTA & Edge: Vision Transformers, SAM 2 adaptations via efficient Low-Rank Adaptation (LoRA) (Coming soon)
## Key features:

Unified Object-Oriented API: Every model inherits from a clean ModelBase class supporting standard .train(), .eval(), and .help() workflows.

Native LoRA Wrappers: Fine-tune massive vision models and transformers on modest hardware (like a single RTX 3070) without blowing up your VRAM. (Coming soonn)

Hugging Face pretrained weights availablle

MLflow Tracking Out-of-the-Box: Keep tabs on your losses, metrics, and hyperparameter logs without writing boilerplate monitoring code.

## Contributions:

Contributions, feature requests, and bug reports are welcome! Feel free to open an Issue or submit a Pull Request on GitHub.

## Author

Sébastien Doyez
ML & AI Engineer,
Interested in Agentics, Computer Vision, Robotics & Deep Learning
