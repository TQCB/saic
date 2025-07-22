# SAIC: Semantic-Aware Image Compression

This repository presents **SAIC (Semantic-Aware Image Compression)**, an image compression framework that leverages semantic saliency to achieve efficient and perceptually optimized compression. The system integrates modules for understanding semantic importance within an image and then spatially compressing the image based on this saliency.

## How it Works

SAIC comprises two main modules: the **Semantic Saliency Module** and the **Spatial Compression Module**.

### 1. Semantic Saliency Module

This module identifies and quantifies the "saliency" or importance of different regions within an input image based on textual prompts.

*   **Semantic Understanding (Grounding DINO)**:
    *   An input image and text prompt are fed into Grounding DINO.
    *   Grounding DINO performs object detection and grounding, generating bounding boxes around semantically relevant areas.
*   **Segmentation (SAM)**:
    *   The bounding boxes from Grounding DINO are then used by SAM (Segment Anything Model).
    *   SAM generates a precise saliency map, which highlights the important regions identified by the previous step.

### 2. Spatial Compression Module

This quantized auto-encoder module handles the actual compression and decompression of the image, guided by the generated saliency map.

#### Encoding Process

1.  **Encoder**:
    *   The original image and the saliency map are processed by the AE encoder.
    *   This encoder transforms the image and saliency map into latent representations.
2.  **Saliency-Guided Quantizer**:
    *   The latent saliency map is used to calculate a dynamic quantization step size $\Delta(i,j) = \frac{\Delta_{base}}{1+\beta\times M_{latent}(i,j)}$. This formula ensures that important (saliency-dense) regions have a smaller $\Delta$ (finer quantization), while less important regions have a larger $\Delta$ (coarser quantization).
    *   The latent image is then quantized using this saliency-guided step size: $round(\frac{x(i,j)}{\Delta(i,j)})$.
3.  **Bit Encoding (Entropy Coder)**:
    *   The quantized latent image is passed to an entropy coder that generates a space-efficient bitstream that we can store.

#### Decoding Process

1.  **Entropy Decoder**:
    *   To reconstruct the image, the compressed bitstream is first fed into an entropy decoder that reconstructs our quantized latent image.
2.  **Decoder**:
    *   The quantized latent image $\hat{y}$ is then passed to the AE decoder that reconstructs the image $\hat(x)$.

## Architecture Diagram

The architecture and data flow of the SAIC system can be visualized in the following diagram:

![Mermaid Diagram](assets/architecture.svg)

## Features

*   **Semantic-aware Compression**: Integrates state-of-the-art semantic understanding models (Grounding DINO, SAM) to identify regions of interest.
*   **Perceptually Optimized Quantization**: Utilizes a saliency-guided quantization strategy to allocate more bits to perceptually important areas, improving visual quality.
*   **Efficient Encoding**: Employs an entropy coder for compact bitstream generation.
*   **Reconstructive Decoding**: Provides a full pipeline for decoding and reconstructing the compressed image.