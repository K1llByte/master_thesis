# Meeting

<!-- Questions -->
<!--
- Is it worth to include the times of the inverse fft? The difference of forward and inverse is negligible
    - Prove that it is negligible and then there's no need to include it
- Besides the graphs ill have to analyse differences between implementations, can i have a table with all relevant numeric results?
    - Include benchmark table in the apendix
- Should I include in-place cuFFT data too (its not the MOST performant)?
    - Mention why we only benchmark out-of-place
- In lists should i use '.' or ';'?
    - ';' com '.' na ultima
- In 6.1 should i include code snipet of cuFFT in apendix?
    - Yes
-->

<!-- Notes -->
<!--


"is presented in" instead of such as

Datas de entrega:
- 31 Dec

Todo[Content]:
- [ ] Corrigir informaçao sobre que o warp size deve ser por volta de 32

Todo[Fixes]:
- For every chapter and section, write a brief description of what it contains with references for the parts
- Label every chapter, section, subsections and figures and all that
-->


## Goals

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
        - [x] 5.1 GPU Programming model 
        - [x] 5.2 2D Fourier Transform on the GPU
        - [ ] 5.3 Implementation Analysis in GLSL
            <!-- - [ ] 5.3.1 GLSL  -->
            - [ ] 5.3.2 Implementation
        - [ ] 5.4 Case of study
            - [ ] 5.4.1 Tensendorf waves
            - [ ] 5.4.2 Implementation details
            - [ ] 5.4.3 Results
    - [ ] 6 ANALYSIS AND COMPARISON
        - [ ] 6.1 cuFFT
        - [ ] 6.2 Implementation analysis in GLSL
        - [ ] 6.3 Implementation analysis in CUDA
    - [ ] 7 CONCLUSIONS AND FUTURE WORK

### Practical

___
<!--
# 4 COMPUTATION OF THE FOURIER TRANSFORM
## 4.1 Improving the Cooley-Tukey algorithm
### 4.1.1 Natural order Cooley-Tukey
### 4.1.2 Stockham algorithm
### 4.1.3 Radix-4 instead of Radix-2
- Brief introduction to multiple factorizations
- Theres pros and cons to higher radix factorizations
- Mention there can be mix-radix implementations and that radix4 can be combined with radix2 easily to allow
to compute the same sizes
- Explain with dragonfly the main difference between stockham radix-2 and 4
- 
-->

# 5 IMPLEMENTATION ON THE GPU <!-- (or) GLSL, both titles apply -->
<!-- FIXME: Maybe change this title to GPU architecture -->
<!-- Programming model information will be mentioned in the implementation details in GLSL -->
## 5.1 GPU Programming model 
> Note: Since the readers might not be within HPC then i
need to introduce some roots to justify implementations

- GPU architecture brief
<!-- - GPU programming model brief -->

## 5.2 2D Fourier Transform on the GPU
- 2D FFT
- Two pass approach for Horizontal and Vetical passes
<!-- - Memory layout -->

## 5.3 Implementation Analysis in GLSL
- Matrix transposition mention

### 5.3.1 GLSL

- Brief of GLSL (what it is and how it is used)
- Why we're using it

### 5.3.2 Implementation

- Say it was an iterative process by applying, studying and testing
- 2D fft computes a lot of 1d fft's so each performancce improvement in the algorithms will be noticeable
- Explain every iteration of the FFT implmentations and GLSL and GPU programming good practices

<!--
#### Cooley-Tukey
- most naive implementation
- Mention to use GLSL bitreverse instead of manual
- pass per stage
    - The way it is dispatched and why it is made that way
- Updating to all stages in a single pass
    - One problem of this is the synchronization between threads

#### Radix-2 Stockam
- Why there are benefits on using stockham on the GPU
    - No bit reversal step

#### Radix-4 Stockam
- How the size of the kernel is affected by this and the performance acquired
- How there are less synchronization
- Why not higher radices? Cons of size constraints and portability to more GPUs
- Performance of higher radices depends on the hardware



-->

<!--
Say:
- In this thesis we provide a set of experiments that effectively study key components in the implementation of FFT's that
matter and impact the performance
-->


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

<!-- #### Setup
- Implementation setup
    - Using Nau3D engine with 2 passes and 2 pingpong image buffers
- What application it was tested
    - The input image was sampled as a texture
    - 2D simple Forward FFt and Inverse FFT display mipmapped -->

# 6 ANALYSIS AND COMPARISON
- Explain whats this chapter about
- First we'll make the comparison with same algorithms in CUDA
- Then we'll talk about popular implementations and the merit they have and then compare them with our implementation in GLSL

## 6.2 cuFFT
- What it is
- Pros and cons (NVIDIA only)
- Sample code for in and out of place implementations
- It selects the best algorithm and properties of the fft to achieve best performance
- The performance of cufft is used as a reference for the difference of the other benchmarks

## 6.1 Implementation analysis in GLSL
- Compare all implementations from section 5.3.2
- Details reasons why the results are this way 

## 6.3 Implementation analysis in CUDA
- Brief the comparisons
- Explain the setup of the comparisons and how does it differ from the GLSL implementation so that the results can be justified
Mention benchmark method (cuda events over the default stream)
- Explain how the similar implementations scale differently and this is worth noting when implementing fft for a specific platform
- Attach results and graphs and reflect

<!--
## 6.1 Popular implementations
- Talk about popular implementations
- FFTW but since we care about GPU oriented approaches we use as reference cuFFT

### 6.1.1 cuFFT
- What it is
- Pros and cons (NVIDIA only)
- Sample code for in and out of place implementations

### 6.1.2 Environment overhead

## 6.2 Comparison with CUDA implementation
- Brief the comparisons

### 6.2.1 CUDA Setup
Explain the setup of the comparisons and how does it differ from the GLSL implementation so that the results can be justified
Mention benchmark method (cuda events over the default stream)

### 6.2.2 Results
- attach all results and graphs
-->

# 7 CONCLUSIONS AND FUTURE WORK