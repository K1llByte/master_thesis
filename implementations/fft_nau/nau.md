# Nau

## 2D FFT for 256px image project

### Notes:
- Since the vert + frag shaders operate through a quad the only vertex shader there is only passes the positions to the fragment shader, which will to something to the texture
- To display the image the pixels must have the log of magnitude of the complex results
- Bit reversal step (can be constexpr)
- Number of dynamic parameters (uniform variables) can be reduced
- The 2D algorithm is FFT DIT
- IMPORTANT NOTE: Reimplement with Compute shaders but with vec4 (2 butterflies at a time) to improve SIMD performance

### Stages

(original) -[Convert to Complex]-> ()

- Convert to Complex (can be constexpr) 
    > Takes input texture with 3 int as rgb values (can also accept 1 greyscale value) and converts them to complex, which is 2 float, one for real and other for imaginary part. When input is rgb then the real value will be the relative luminance (https://en.wikipedia.org/wiki/Relative_luminance)
    - INPUT: RGB_F32 | R_F32
    - OUTPUT: RG_F32

- Horizontal Forward FFT
    > Compute 1D FFT for every row

- Vertical Forward FFT
    > Compute 1D FFT for every column

- Copy Texture
    > Although this copies the texture to a new one it actually converts to log of magnitude which is log2(sqrt(r^2 + i^2)) for every rgb value

<!--
- Horizontal Inverse FFT
- Vertical Inverse FFT
-->