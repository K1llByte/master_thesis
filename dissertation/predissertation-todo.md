# Index

- [Todo](#todo)
- [Notes](#notes)
    - [Fancy Words](#fancy-words)
    - [Tips](#tips)
    - [Dissertation Topics](#dissertation-topics)

# Aux
<!-- TODO: Reorganize this -->
- State of the art
<!-- Introductory section where FFT is explained and its implementations -->
<!-- . . . -->
<!-- FFT explanation -->
    - Fast Fourier Transforms
    - Radix-2 Decimation-in-Time FFT
    - Radix-2 Decimation-in-Frequency FFT
    
<!-- NOTE: Im not sure if i should have this topic ^ right now -->


- Comparison of implementations {volatile name, maybe GPU FFT's?}
<!-- This section will provide insight of the performance of different implementations and how to implement FFT efficiently -->
    - CUDA implementation
    - cuFFT
    - GLSL (Compute Shaders)

- Cases of study (application implementations) {volatile name}
<!-- This section will analyse and justify computation of fourier transforms to real applications -->
    - Image filtering
    - Computer graphic Waves

<!-- === -->

# Todo

- Optimize: Only power of 2, 2D transforms
- GPU Buffer vs Texture
- Layout transitions opt
- Compute shaders
- Fragment shader


- [ ] List the main objectives and goals for this dissertation
    - Provide snipets of code with reference implementations
    - Optimize/specialize implementations of applications with FFT
    - Benchmark different implementations and stratagies
    - Compare equivalent FFT computations with different GPU frameworks (CUDA and GLSL)


- [1/2] *Discrete Fourier Transforms*
    - Example implementation in python of a DFT
    - ITS MISSING SOME PHRASES SPEAKING ABOUT THE REPRESENTATION OF THE MAGNITUDE AS THE EUCLIDEAN DISTANCE OF THE COMPLEX COORDINATES, AND ALSO TALK ABOUT PHASE (might not be needed since might be too specific)
- [ ] *Fast Fourier Transforms*
    - [ ] Radix-2 Decimation-in-Time FFT
        - Algorithm specification
        - Reference implmentation
    - Radix-2 Decimation-in-Frequency FFT
      [ ]   - Algorithm specification
        - Reference implmentation
    - There's also 2d and 3d fft's

- [ ] *Computation of FFT*
    - Mention that this algorithm is used as a benchmark in parallel computers due to its hard nature
    - Computation of this algorithm on the CPU vs GPU
    - Example of implementation in python of FFT (compare with DFT)
        - Mention this is CPU computation
        - Mention that this is just an implementation to provide a simple algorithmic description of the  computation of the Cookley... FFT
    - NOTE: Have in mind that the computation of the FFT as a general FT is very different fro ma specialized application case, so a lot of algorithms that take into account some properties they want for the computation might not be required (Large input, non-power f 2, real values, some or no precomputation of values (as textures))

- [ ] *Related Work*
    - [ ] cuFFT
    - [ ] (microsoft article)

- [1/2] Extra: Ponder making an extra chapter with nomenclature (Acronims)
    - FFT
    - DFT
    - IFFT* -> Inverse Discrete Fourier Transform
    - FFFT* (or just FFT) -> Forward Fast Fourier Transform
    - IFFT* -> Inverse Fast Fourier Transform

- After finishing the State of the art:
    - [x] *Document Organization*
    - [1/2] Finish *Objectives*
        - Mention that this dissertation is focused on the optimization of computation of fft so the
        fourier transform theory wont be proven and we wont go on a deep explanation on the how it works
    - [1/2] *Motivation*
    - [1/2] *Contextualization*
        - Say that its required basic understanding of complex numbers and linear algebra

## Notes

- Every equation name must be referenced by number, the name is named in the explanation
- Reference Figure when context is needed, and reference the author
- Every equation name must be referenced by number, the name is named in the explanation

### Fancy Words
> **IMPORTANTE:** NAO ESQUECER DE INVOCAR REFERENCIAS  
> **FANCY WORDS:**
> - aforesaid - mencionado anteriormente/acima
> - ubiquitous - onipresente
> - empirical - 'based on, concerned with, or verifiable by observation or experience rather than theory or pure logic. ("they provided considerable empirical evidence to support their argument")'
> - culminate -  'to reach the highest point, summit, or highest development'
> - Hence - portanto
> - festinating - to hasten, to hurry (means hurried)
> - cogitating - to spend time thinking very carefully about a subject. (thinking)

### Tips

#### Consistency
```latex
\label{chap:foo-bar}
\label{sec:foo-bar}
\label{subsec:foo-bar}
\label{fig:foo-bar}
\label{eq:foo-bar}
\label{alg:foo-bar}
```

### 
- 0.91ms p/ frame
- 1.47 vert pass
- 1.23 horizontal pass
- 512 sequence size


António Silva
Carlos Brito

### Dissertation Topics

<!-- =============== TEMPLATE =============== -->

**Template**
```md
# COPYRIGHT AND TERMS OF USE FOR THIRD PARTY WORK
# ACKNOWLEDGEMENTS
# STATE OF INTEGRITY
# ABSTRACT
# RESUMO
# CONTENTS
# LIST OF FIGURES
# LIST OF TABLES
# LIST OF LISTINGS
## Part I - Introductory Material
### 1. Introduction
### 2. State of the Art
### 3. The problem and its challenges
## Part II - Core of Dissertation
### 4. Contribution
### 5. Applications
### 6. Conclusion and future work
## Part III - Apendices
### A. Support work
### B. Details and results
### C. Listings
### D. Tooling
```

<!-- ================= MINE ================= -->

**Mine**
```md
# COPYRIGHT AND TERMS OF USE FOR THIRD PARTY WORK*
# ACKNOWLEDGEMENTS*
# STATE OF INTEGRITY*
# ABSTRACT
# RESUMO
# CONTENTS*
# LIST OF FIGURES*
# LIST OF TABLES*
# LIST OF LISTINGS*
## Part I - Introduction
### Contextualization
### Motivation
### Objectives [1/2]
### Document Organization
## Part II - State of the Art
### Fourier Transform
#### What is Fourier Transform
#### Where it is used
### Discrete Fourier Transform
#### Matrix Formulation
### Fast Fourier Transform
#### Radix-2 Decimation-in-Time FFT
#### Radix-2 Decimation-in-Frequency FFT
### Related Work
#### cuFFT
#### (microsoft article)

<!-- Final work will have the Part III - Core of Dissertation -->
```
