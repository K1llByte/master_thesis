> **IMPORTANTE:** NAO ESQUECER DE INVOCAR REFERENCIAS  
> **PALAVRAS CHIQUES:**
> - aforesaid - mencionado anteriormente/acima
> - ubiquitous - onipresente
> - empirical - based on, concerned with, or verifiable by observation or experience rather than theory or pure logic. ("they provided considerable empirical evidence to support their argument")
> - culminate -  to reach the highest point, summit, or highest development

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
#### Discrete Fourier Transform
### Fast Fourier Transform
#### Computation of FFT
#### cuFFT
### Related Work
<!-- Final work will have the Part III - Core of Dissertation -->
```

___
# Abstract
Foobar
# Resumo
Foobar mas em português

___
## Part I - Introduction
### Objectives

<!--
**Objetivos de alto nivel:**

1. Explicar em um primeiro paragrafo o que quero com esta dissertação:
- Explorar os graus de performance de fft em aplicações em concreto, comparando ferramentas dedicadas para alta performance desses algoritmos, com implementações especializadas

Informar:
- que esta dissertação irá focar em implementaçoes de fft baseadas no algoritmo de divide and conquer de Cooley–Tukey 

**Objetivos de baixo nivel / O que vamos fazer em concreto:**

e que um dos principais objetivos vai ser otimizar a utilização destes algoritmos em tempo real nos casos de estudo, que vao estar principalmente no dominio de computação gráfica (apesar de haver diversas outras áreas qe podiam ser estudadas)

Similarly, many studies have their work focused on the optimized computation of the FFT
-->

The main objective of this dissertation is to provide efficient FFT alternatives in GLSL compared with dedicated tools for high performance of FFT computations like NVIDIA cuFFT library, while analysing the intrinsic of a good Fast Fourier Transform implementation on the GPU.
To accomplish the main objective there are two stages taken in consideration, "Analysis of CUDA and GLSL kernels" to be well settled in their differences and to have a reference for the second stage "Analysis of cuFFT and GLSL FFT" which will cluster the study's main objective.

To compose a final verdict conclusion, we will use as case of study applications with implementation of the FFT in the field of Computer Graphics that require realtime performance.


## Part II - State of the Art
### Fourier Transforms

<!-- What is Fourier Transform -->

The **Fourier Transform** is a mathematical method to transform the domain refered to as *time* of a function, to the *frequency* domain, intuitively the Inverse Fourier Transform is the corresponding method to reverse that process and reconstruct the original function from the one in *frequency* domain representation.

Although there are many forms the Fourier Transform can assume the key definition can be described as:

<!-- Forward Fourier Transform -->
X(\omega) = \int_{-\infty}^{+\infty} x(t)e^{-i \omega t} dt
<!-- Inverse Fourier Transform -->
x(t) = \frac{1}{2\pi} \int_{-\infty}^{+\infty} X(\omega)e^{-i \omega t} d\omega

- x(t) -> function in *time* domain representation  
- X(\omega) -> function in *frequency* domain representation  
- i -> imaginary unit i = \sqrt{-1}  

<!-- TODO: Onde é usado -->

This model of the fourier transform applied to infinite domain functions is called **Continous Fourier Transform** and its targeted to the calculation of the this transform directly to functions with only finite discontinuities in x(t). <!-- (|x(\alpha+)-x(\alpha-)|<\infty ) -->

<!-- Discrete Fourier Transform -->

The Fourier Transform of a finite sequence of equally-spaced samples of a function is the called the **Discrete Fourier Transform** (DFT), it converts a finite set of values in *time* domain to *frequency* domain representation. Its the most important type of transform since it deal with a discrete amount of data, which can be implemented in computers and be computed by specialized hardware.

<!-- Forward Discrete Fourier Transform -->
X{\scriptstyle k} = \sum_{n=0}^{N-1}x{\scriptstyle n} \cdot e^{- \frac{i 2 \pi}{N}kn}
<!-- Inverse Discrete Fourier Transform -->


<!-- "computes to the domain of complex numbers" -->

<!-- - Discrete Fourier Transform -->
<!-- - Continuous Fourier Transform -->
<!--
1. Explicar o que é Fourier Transform
    - O que é
    - Para que serve
    - Definição
    - Onde é usado
    - isto eventualmente vai mencionar o que são Continuous Fourier Transforms e Discrete Fourier Transforms
2. Explicar de que forma Fourier Transforms são uteis e aplicadas no mundo real
3. Começar a falar de DFT e reparar que a solução é O(N^2)
-->