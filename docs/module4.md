### Q1: RGB (no learning)
1. **Lab colour**: distances there track perceived colour difference.
2. **Background model**: pixels in a border band (or outside your box) are clustered into a few
   colours with k-means; each cluster keeps a mean and covariance.
3. **Novelty**: each pixel's Mahalanobis distance to the nearest background cluster.
4. **Hysteresis**: pixels above *high* are seeds; pixels above *low* join if connected to a seed.
5. **Morphology + largest component + hole filling** give a coarse mask.
6. **Watershed** with sure-foreground (eroded mask) and sure-background (outside dilated mask)
   markers snaps the outline to the strongest nearby edges.

### Q2: Thermal
A person is usually warmer than the background: CLAHE to even out contrast, Otsu threshold,
opening/closing, largest component, then the same watershed refinement.

### Comparing with SAM2
Same image, same box prompt. IoU = |A∩B| / |A∪B|; Dice = 2|A∩B| / (|A|+|B|). Boundary F-score:
precision and recall of boundary pixels within a tolerance (≈0.75% of the diagonal), using distance
transforms. Hausdorff distance is the worst boundary disagreement.

### Q3: Fourier-domain edges and regions
- **Derivative theorem**: 𝓕{∂f/∂x} = j2πu F(u, v). Multiplying by j2πu and j2πv (times a Gaussian
  to suppress noise) and inverting gives the gradient; its magnitude marks edges.
- **Laplacian of Gaussian**: 𝓕{∇²(G∗f)} = −4π²(u²+v²) Ĝ F. Edges are its zero crossings.
- **High-pass**: (1 − Ĝ) F keeps edges and texture.
- **Regions**: a low-pass Ĝ F keeps large uniform areas; thresholding it (Otsu) segments regions.
- Ideal (sharp cut-off) filters ring, since their spatial kernel is a sinc; Gaussians don't.
