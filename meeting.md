# Meeting

<!-- Duvidas -->
<!--

-->

<!-- Notes -->
<!--
2X Out
31 Dec

Reduzir info em 5.1
[i] Prioridade resultados praticos (only missing benchmarks in tensendorf waves)
[x] Resultados CUDA
[x] Resultados Stockham Radix-4

Enviar ao orientador blocks de resultados, todos de glsl, todos de cuda, todos de cufft etc ...
-->


## Goals

- [] Results
- [1 month] Write until chapters 4 and 5 (incomplete stuff for the last pratical phase, algos, code blocks, etc ...) []
- [2 weeks] Final practical corrections ...

### Writing

- Correct written practices: (Only at the end)
    - When talking about the work done remove the "We" and reference it as "It was done"
    - "listings" on the first page of the pre-dissertation stuff
    - Add/Update Keywords
- Review of the written text

<!--
x - Done
i - incomplete, because needs something from the final pratical phase
-->

> Note: Every topic title is volatile so they can be reorganized and changed

- Content topics:
    - [i] 4 COMPUTATION OF THE FOURIER TRANSFORM
        - [x] 4.1 Improving the Cooley-Tukey algorithm
            - [x] 4.1.1 Natural order Cooley-Tukey
            - [x] 4.1.2 Stockham algorithm
            - [i] 4.1.3 Radix-4 instead of Radix-2
    - [i] 5 IMPLEMENTATION ON THE GPU 
        - [i] 5.1 GPU Programming model 
        - [ ] 5.2 2D Fourier Transform on the GPU
        - [ ] 5.3 Implementation Analysis in GLSL
            <!-- - [ ] 5.3.1 GLSL  -->
            - [ ] 5.3.2 Implementation
        - [ ] 5.4 Case of study
            - [ ] 5.4.1 Tensendorf waves
            - [ ] 5.4.2 Implementation details
            - [ ] 5.4.3 Results
    - [ ] 6 ANALYSIS AND COMPARISON
        - [ ] 6.1 Popular implementations
            - [ ] 6.1.1 cuFFT
            - [ ] 6.1.2 FFTW
        - [ ] 6.2 Comparison with GLSL implementation
            - [ ] 6.2.1 Results
    - [ ] 7 CONCLUSIONS

### Practical

- Final stage:
    - Corrections on the algorithms
    <!-- - OpenCL impl -->
    - Rest of the benchmarkings
        - And radix-4
        - for CUDA vs GLSL overhead comparison)

___
<!--
# 4 COMPUTATION OF THE FOURIER TRANSFORM
## 4.1 Improving the Cooley-Tukey algorithm
### 4.1.1 Natural order Cooley-Tukey
### 4.1.2 Stockham algorithm
### 4.1.3 Radix-4 instead of Radix-2
-->

# 5 IMPLEMENTATION ON THE GPU <!-- (or) GLSL, both titles apply -->
## 5.1 GPU Programming model 
> Note: Since the readers might not be within HPC then i
need to introduce some roots to justify implementations

- GPU architecture brief
- GPU programming model brief

## 5.2 2D Fourier Transform on the GPU
- 2D FFT
- Two pass approach for Horizontal and Vetical passes
- Memory layout
- Matrix transposition mention

## 5.3 Implementation Analysis in GLSL

### 5.3.1 GLSL

- Brief of GLSL (what it is and hwo it is used)
- Why we're using it

### 5.3.2 Implementation
- Explain every iteration of the FFT implmentations and GLSL and GPU programming good practices

## 5.4 Case of study
### 5.4.1 Tensendorf waves
- Mention it is a good example to benchmark since the fft takes a big role into play (2 ffts multiple times) and effects most of the performance

### 5.4.2 Implementation details
- Explain the old implementation (and explain that it is an example of generic cooley-tukey pass per stage fft)
- Explain how it was implemented a better version using previous concepts

### 5.4.3 Results
- Compare the results of the previous implementation with the new implementation
- Possibly use many comfigurations for more deep testing
- Preview of the rendered image and graphs with the improvements results


# 6 ANALYSIS AND COMPARISON
## 6.1 Popular implementations
- Talk about popular implementations
- FFTW but since we care about GPU oriented approaches we use as reference cuFFT

### 6.1.1 cuFFT
- What it is
- Pros and cons (NVIDIA only)
- Sample code for in and out of place implementations

### 6.1.2 OpenCL
- What it is
- Pros and cons
- Sample code for in and out of place implementations

## 6.2 Comparison with GLSL implementation
- Brief the comparisons
### 6.2.1 Results
- attach all results and graphs

# 7 CONCLUSIONS


___
# Meeting 2

<!-- Duvidas -->
- Há algum novo template?