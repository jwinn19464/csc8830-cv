### Optical flow and what it tells you
Farneback flow gives a vector per pixel. A robust (RANSAC) similarity fit to the whole field is the
**camera motion**: translation = pan, scale = moving forward/zooming, angle = roll. Pixels that
disagree with that fit are **independently moving objects**. When the camera moves forward, flow
lines radiate from the **focus of expansion**, found by least squares: each vector f at point p gives
the line n·(x − p) = 0 with n ⟂ f.

### Lucas-Kanade
Brightness constancy I(x+u, y+v, t+1) = I(x, y, t); first-order Taylor: Iₓu + I_y v + I_t = 0.
One equation, two unknowns, so assume (u, v) is constant in a window and solve least squares:

[ΣIₓ² ΣIₓI_y; ΣIₓI_y ΣI_y²] [u v]ᵀ = −[ΣIₓI_t ΣI_yI_t]ᵀ

The 2×2 matrix must be well conditioned (both eigenvalues large: a corner). Iterate by warping
frame t+1 with the current estimate; use an image pyramid for large motion.

### Bilinear interpolation
For x₀ = ⌊x⌋, a = x − x₀, b = y − y₀:
I(x, y) = (1−a)(1−b)I₀₀ + a(1−b)I₁₀ + (1−a)b I₀₁ + ab I₁₁. It's needed because tracked
positions are sub-pixel.

### Validation
Our tracker is compared point by point with OpenCV's, and the mean absolute patch difference at
the predicted location is compared with that at the unmoved location (it should drop).

### Structure from motion for a planar object
With K known, the homography from view 1 to view k decomposes as H ≃ K(R + t nᵀ/d)K⁻¹. The
decomposition with the plane in front of camera 1 and the same normal n in every view is kept.
Points are placed by ray-plane intersection X = d·m / (n·m), m = K⁻¹x̃, with d = 1. Bundle
adjustment then refines all poses and points to minimise reprojection error in all four views.
One known length sets metric scale, and the tapped corners give the object's dimensions.
