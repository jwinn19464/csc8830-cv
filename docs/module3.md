### Spatial convolution
g(x, y) = Σᵢ Σⱼ h(i, j) f(x − i, y − j). The code loops over the kernel taps and adds shifted
copies of the image, so it is exact and still fast.

### Fourier domain
Convolution theorem: 𝓕{f ∗ h} = 𝓕{f} · 𝓕{h}, so g = IFFT( FFT(f) · FFT(h_padded) ). The kernel is
zero-padded to the image size and its centre is rolled to pixel (0, 0); otherwise the result is
shifted by half the kernel.

### Why they match exactly
The DFT treats the image as periodic, which makes its product a **circular** convolution. Using
wrap-around edges in the spatial version gives identical boundaries, so the two agree to
floating-point precision (MSE around 10⁻²⁵). With zero or reflect padding they'd differ only near
the borders.

### What sigma does
A Gaussian of std σ in space is a Gaussian of std 1/(2πσ) in frequency: bigger σ keeps fewer
high frequencies. The kernel has to be about 6σ wide to hold the Gaussian; if it's much smaller,
the truncated kernel is close to a box filter, whose spectrum is a sinc with ringing side lobes.

### Cost
Spatial: O(N·k²). FFT: O(N log N), independent of kernel size, so it wins for large kernels.
