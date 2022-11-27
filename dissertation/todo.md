# Index

- [Todo](#todo)
- [Notes](#notes)
    - [Dissertation Topics](#dissertation-topics)

## Todo

<!-- Today -->
<!-- Dissertation -->
- Organizar os topicos que vao ser escritos no futuro
- Começar a escrever primeiro topico
- Get more references
- Adicionar "OTFFT: High Speed FFT library" como referencia
<!-- Pratical -->
- Fazer uma implementação que usa FFTW
- Fazer uma implementação que usa radix-4
<!-------------->
<!-- Dissertation -->
<!-- Pratical -->

## Notes

### Dissertation Topics

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
<!-- A topic dedicated to the most well known Cooley-Tukey algorithm -->
<!-- TODO: Mention that this algorithms are usually showcased in its recursive form
however the iterative version showcases less call instructions overhead -->
#### Radix-2 Decimation-in-Time FFT
#### Radix-2 Decimation-in-Frequency FFT


## Part III - Computation of the Fourier Transform
### Improving the Cooley-Tukey algorithm
<!--
How this algorithm benefits compared to the cooley-tukey one
-->
<!-- Explore inline algorithms -->
#### Natural order Cooley-Tukey
<!--
No bit reversal step
It isnt mentioned enough on fast search implementation
Improves the performance radically
-->
#### Stockham algorithm
#### Radix-4 instead of Radix-2
<!-- Restrictions -->
<!-- Show some study's benchmark of Radix-4 vs Radix-2 -->

### Bidimentional FFT
<!-- Explain how to compute the 2D FFT of an image and showcase an example -->

## Part IV - Implementation on the GPU
### Fourier Transform on the GPU
<!--
Introduce GPU programming, implications and what drives the motivation to
change the cooley-tukey algorithm to get more performance.
- Streamed computations (could be an optimization)
-->
### Implementation in GLSL
<!--
How did i organize the implementation to be able to sync the local threads (A Work Group takes ownership of an entire row of the image and serveral local threads work on parts of the row, syncing each other for every stage)
-->

## Part IV - Analysis and Comparison
### Popular implementations
#### cuFFT
<!-- Explain configuration options and usage -->
#### FFTW
### Comparison with GLSL implementation
#### Analysis
<!--
Exaplain here why there's a macro for multiple butterflies and explain it didn't work for sizes to 4096
Pros: no external dependencies, not bound to the library target devices support
-->

### Related Work
#### cuFFT
#### (microsoft article)
```