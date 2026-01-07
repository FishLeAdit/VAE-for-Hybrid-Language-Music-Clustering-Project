# Unsupervised Music Clustering Using Variational Autoencoders and Hybrid Audio–Lyrics Representations

This project investigates **unsupervised music clustering** using learned latent representations from **Variational Autoencoders (VAEs)** and **convolutional autoencoders**, with a focus on **hybrid audio–lyrics feature fusion**.  
We compare classical handcrafted audio features with deep representation learning approaches and evaluate clustering quality using standard unsupervised metrics.

---

## Project Objectives

- Learn meaningful low-dimensional representations of music audio using VAEs
- Explore whether **deep latent spaces** are more cluster-friendly than handcrafted features
- Incorporate **lyrics information** via text embeddings and study multi-modal fusion
- Compare clustering performance across:
  - MFCC + PCA baselines
  - Dense VAE
  - Convolutional VAE
  - β-VAE variants
  - Convolutional Autoencoder
  - Audio–Lyrics fused embeddings

---

## Dataset

- **Audio dataset:** Subset of the Free Music Archive (FMA)
- **Tracks used:** ~4,182 English-language songs
- **Audio format:** MP3
- **Lyrics source:** Whisper-transcribed text files
- **Genres:** Multiple high-level genres (used only as proxy labels for evaluation)
## DATASET LINK: https://github.com/mdeff/fma

All models are trained in a **fully unsupervised** manner.

---

## Feature Extraction

### Audio Features
- **MFCCs (baseline):**
  - 40 coefficients per track
  - Aggregated via temporal averaging
- **Log-mel spectrograms (deep models):**
  - 64 mel bands
  - Fixed temporal length (~215 frames)
  - Input shape: `1 × 64 × 215`

### Lyrics Features
- Whisper-generated lyric transcripts
- TF–IDF vectorization
- Vocabulary learned from the full lyrics corpus

### Fusion
- Early fusion via **concatenation**
- Audio latents (z-score normalized)
- Lyrics TF–IDF vectors (L2 normalized)

---

## Models Implemented

### Representation Learning
- **Dense VAE**
  - Input: MFCC vectors
  - Latent dimension: 8
- **Convolutional VAE**
  - Input: log-mel spectrograms
  - Latent dimension: 16
- **β-ConvVAE**
  - β values: {1, 4, 10}
  - Latent dimension: 16
- **Convolutional Autoencoder**
  - Deterministic baseline
  - Latent dimension: 16

### Training Setup
- Optimizer: Adam
- Learning rate: `1e-3`
- Batch size: `32`
- Epochs: `30–50` (depending on model)
- Hardware: CPU (GPU optional)

---

## Clustering Methods

Clustering is applied **after representation learning**.

- **K-Means**
  - k ∈ {2, 3, 4, 5}
- **Agglomerative Clustering**
  - Ward linkage
- **DBSCAN**
  - Density-based clustering (fixed ε and min samples)

---

## Evaluation Metrics

- **Silhouette Score** (primary metric)
- **Davies–Bouldin Index**
- **Adjusted Rand Index (ARI)**
  - Used cautiously with genre labels as *proxy ground truth*
- **UMAP visualizations**
  - Qualitative inspection of latent structure

---

### How to Run
- First Run the Dataset Processing Scripts according to numerical order, to download the dataset, and fetch lyrics(from genius using genius.api) and to transcribe lyrics using OpenAI Whisper
- Once that's done, run the scripts in SRC in numerical order 

## Credits
This project uses code from
- https://github.com/voidism/mfcc_extractor
- https://github.com/park1996/Music-Classification-By-Genre
- https://github.com/pytorch/examples/tree/main/vae
- https://github.com/QiangSu/VAE-clustering
- https://github.com/yjlolo/vae-audio
- https://github.com/sksq96/pytorch-vae
- https://gist.github.com/stes/92db6023aa3dab5d13e49ece198102c7
- https://github.com/librosa/librosa




