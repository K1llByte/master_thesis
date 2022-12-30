# Meeting

<!-- 
--------------[DUVIDAS ATE ENTREGA]--------------
- Eu vou ir melhorando coisas ate entrega, mas alguma coisa em especifico que me devia focar que seja mais importante?
- Qual a melhor hora para submeter? pergunto por causa 




--------------[PRECISA EXCLARECIMENTO]--------------
- Euler formula, é so usado para a implementação


- Fazer tabelas para resultados e com o ratio
    - Primeiro, selecionar quais sao relevantes de fazer tabelas
    - Consultar o stor sobre essa seleção, e mandar print sobre se encaixam bem ou nao.
- Gráficos que estão lado a lado, junta-los para ter uma noção melhor da diferença (fiz para o case of study)
    - Consultar, porque em algums a escala da jeito para a diferença para o grafico ao lado, mas noutros casos nota-se se é constante/linear
- Dizer para nao apagar o exemplo no Estado de arte, porque dá jeito para mostrar a simetria que se segue no 2 real por 1 complex (Not sure about this)
- Explain the pack and unpack process of the mixed double FFT for the price of 1 (added formulas for packing and unpacking)
    - O unpack acho que nao funciona, as frequencias nao sao simetricas, so o pack é que funciona
    - https://developer.apple.com/documentation/accelerate/data_packing_for_fourier_transforms
- I said that the ocean waves FFTs produce two real values for a complex, so 4*2 2D FFTs
----------------------[TODO]----------------------
- Referencias antigas
    - added > bengtsson2020development
    - old > stuart2011efficient, 
    - referencias de FFT
    - "s, especially for specific hardware (Mermer et al. (2003))."
    - Radix-4 referencias recentes
    - Referencia de cuFFT ser mais rapida mais recente, e varias
- Listing 4.9 dragonfly, reuse code e tal (opcional)
- acronimos no inicio a full e depois so os short
- Update radix-4 code implementation to have the final stage radix2
- Falar na implementação em GLSL da for each loop feature e de como funciona
    - So variaveis constexpr
    - Citar alguma referencia a falar sobre isto
    - Restrições de usar este for loop
--------------[DONE]--------------
- Passar 2D FFT para Fast Fourier Transform
- Merge chapter 2 e 3
- Reais para complexa na parte de computing the FFT
- Nao refenrecias as imagens "The results of the cuFFT out-of-place benchmarks can be found in Figure 13 and ??"
- Detalhar results e lua scripting e CPU time em que consiste
    - Dizer que é motor generico
- Figura 12 está mal apagar ou corrigir
- Remover multiple butterflies
- Figura 13 legenda ou texto indicar que sao as unique pass approaches
- Remover o "é linear"
- Corrigir " despite sizes above 1024, it is"
- Mudar tabela 2 para ter os resultados individuais e com dupla
- Nao recomendar fortementemente o radix-4
    - Remover radix-2 "is simpler"
-->

<!-- Questions -->
<!--
------------[Updated]------------
- "compensating for the 70\% kernel size increase" Como/onde devo informar as condições com que avaliamos isto? avaliei com um shader compiler para SPIR-V
-----------------------------------
- CONFIRMAR: Delete example in the state of art?
- É 'Case of study' ou 'Case study'?
- Exclarecimento de comentários ainda nao resolvidos:
    - "O detalhe que puseste na transformação discreta é necessário aqui. Da forma como isto está escrito tem muitos passos em que é preciso "trust me bro" :-)"
    - "Ainda falta um passo, a fugira 3 tal como está se obtém directamente da figura 2" 
        - Duvidas sobre se é a figura 3 e 2 do link ou da tese, e tentar justificar o porque de ter deixado assim
- Perguntar ao professor se é plausivel a justificação da data access locality do stockham
- (MULTIPLE FFTs USING VEC4 AND SIMD OPERATORS) Should I put the code in appendix? not that relevant in my oppinion not an implementation, more like an adaptation for the use case.
- "Não é o mesmo algoritmo mas com outro radix?" da mesma forma que Cooley-Tukey é o mesmo algoritmo que Stockham radix-2.
- "Um ponto importante é a possibilidade de realizar duas transformações de inputs reais com o mesmo número de operações que uma transformação complexa." 
-->

<!-- Notes -->
<!--
- "is presented in" instead of "such as"




Todo[Content]:
- [ ] 5. Justificar a derivação da fórmula de DIT em 2.3.1 e igualmente fazer um equivalente para 2.3.2
- Duvida: É suposto apagar o exemplo?

- [ ] Adicionar notas sobre as inversas no estado de arte e capitulo 4
- [ ] Prove that the difference inverse vs forward times are negligible and that's there's no need to include the inverse

- [ ] Corrigir informaçao sobre que o warp size deve ser por volta de 32
- [ ] (Optional) Include benchmark table in the apendix

Todo[Fixes]:
- For every chapter and section, write a brief description of what it contains with references for the parts
- Label every chapter, section, subsections and figures and all that
- Lists items end with ';' and last item with '.'

- [ ] "listings" on the first page of the pre-dissertation stuff
-->

<!-- Temporary -->
<!--

-->

## Goals

**Last submission day:** 31 Dec


### Writing


<!--
x - Done
i - incomplete, because needs something from the final pratical phase
-->

> Note: Every topic title is volatile so they can be reorganized and changed

- Content topics:
    - 2 THE FOURIER TRASFORM
        - 2.1 Continuous Fourier Transform
        - 2.2 Discrete Fourier Transform
            - 2.2.1 Matrix multiplication 
        - 2.3 Fast Fourier Transform
            - 2.3.1 Radix-2 Decimation-in-Time FFT 
            - 2.3.2 Radix-2 Decimation-in-Frequency FFT
        - 2.4 Stockham algorithm
        - 2.5 Radix-4 instead of Radix-2
        - 2.6 Two real inputs within one complex input 
        - 2.7 2D Fourier Transform
    - 3 IMPLEMENTATION ON THE GPU 
        - 3.1 Cooley-Tukey
        - 3.2 Radix-2 Stockham
        - 3.3 Radix-4 Stockham
    - 4 ANALYSIS AND COMPARISON
        - 4.1 cuFFT
        - 4.2 GLSL implementation results
        - 4.3 Case of study
            - 4.3.1 Tensendorf waves
            - 4.3.2 Results
    - 5 CONCLUSIONS AND FUTURE WORK

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

## 6.4 Case of study
- [x] Explain that since the results are direct benchmarks of the FFT performance of converting a 2D image to frequency domain, that we felt the need to improve an implementation of a real use case that heavily relies on FFT to analyse the impact of a good FFT implementation.
- [x] Say the application we choose was realistic ocean rendering using tensendorf waves.

### 6.4.1 Tensendorf waves
- Mention it is a good example to benchmark since the fft takes a big role into play (2 ffts multiple times) and effects most of the performance

- Explain the old implementation, as it corresponds to the starting point we mentioned in the Implementation section 5.3.1 (cooley-tukey pass per stage FFT)
- Write the total number of FFT's the app requires per frame
- Explain how it the FFT pass has to compute multiple at a time

### 6.4.3 Results
- Compare the results of the previous implementation with the new implementation
- Possibly use many comfigurations for more deep testing
- Preview of the rendered image and graphs with the improvements results

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