### Calibrated stereo, depth first
After stereo calibration (K₁, K₂, R, T) both images are rectified so corresponding points share a
row. Disparity d = u_L − u_R, and by similar triangles

Z = f B / d,  X = (u_L − c_x) Z / f,  Y = (v − c_y) Z / f

so each tapped point gets its depth first, then its 3-D position, and sizes are 3-D distances.
Depth error grows as Z²/(fB): ΔZ ≈ Z² Δd / (f B). A longer baseline helps at long range.

### One phone moved sideways
With K from Module 2, matched features give the essential matrix E = [t]ₓR; recoverPose gives R and
the direction of t, and the measured baseline sets |t| = B.

### Shapes
- **Rectangle**: four corners give side lengths (opposite sides averaged) and area.
- **Circle**: centre plus rim points give the radius.
- **Polygon**: vertices give side lengths, perimeter, and area = ½‖Σ Pᵢ × Pᵢ₊₁‖.

### Uncalibrated stereo (by hand)
Without K and R, T, only F can be found from matches (8-point algorithm, x̃ᵣᵀFx̃ₗ = 0). This gives a
projective reconstruction; metric sizes need extra knowledge (known lengths, right angles, or
self-calibration). Write this derivation by hand for the submission.

### Room dimensions
Disparity from semi-global matching becomes a point cloud (reprojectImageTo3D with Q). RANSAC fits
the dominant planes (walls, floor, ceiling) one after another; planes with parallel normals on
opposite sides give the room's length, width and height as the gap between them. The two videos are
synced by a light flick: the largest jump in mean brightness marks the same instant in both.
