### Calibration (Zhang's method)
A flat checkerboard (Z = 0 in its own frame) maps to the image by a homography
H = K [r1 r2 t]. Because r1 and r2 are orthonormal, each photo gives two constraints on
B = K⁻ᵀK⁻¹:  h1ᵀ B h2 = 0  and  h1ᵀ B h1 = h2ᵀ B h2. Three or more photos solve for B, hence K;
then R, t per photo, then all parameters (including lens distortion k1, k2, p1, p2, k3) are
refined by minimising reprojection error. The RMS of that error is the quality score.

### Measuring from one photo
Pinhole projection after undistortion:  u = fx X/Z + cx,  v = fy Y/Z + cy.
For a flat object facing the camera at a measured distance Z:
X = (u − cx) Z / fx,  Y = (v − cy) Z / fy, so the distance between two image points is

D = Z · √( ((u₁ − u₂)/fx)² + ((v₁ − v₂)/fy)² )

**Error sources:** Z is to the optical centre (inside the phone), not its back; tilt makes the
object's depth vary; tap accuracy of ±1 px is ±Z/fx in size; resolution changes scale K.

### Two cameras
Camera 1 at the origin: x̃₁ ≃ K₁[I | 0] X̃. Camera 2 rotated by R and moved by t:
x̃₂ ≃ K₂[R | t] X̃. Substituting X = Z₁ K₁⁻¹ x̃₁:

x̃₂ ≃ Z₁ K₂ R K₁⁻¹ x̃₁ + K₂ t

The first term is the infinite homography, the second the epipole. Without depth, eliminating
Z₁ gives the epipolar constraint x̃₂ᵀ F x̃₁ = 0 with F = K₂⁻ᵀ [t]ₓ R K₁⁻¹. For points on a plane
nᵀX = d the mapping is the homography x̃₂ ≃ K₂ (R + t nᵀ/d) K₁⁻¹ x̃₁. The fixed parameters are
K₁, K₂ (calibration) and R, t (stereo calibration or a measured mount).
