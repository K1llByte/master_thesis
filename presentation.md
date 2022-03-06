# Presentation

## Titulo do tema

O meu nome é jorge mota e o tema da minha dissertação de mestrado é relativo ao estudo
de desempenho de implementações de transformadas discretas de fourier especializadas em GLSL.

<!-- Alternativamente também pode ser entitulado "Alta performance de Transformadas Rápidas de Fourier no GPU". -->

## Contextualização e Objetivos

A Transformada Rápida de Fourier é dos algoritmos mais importantes para calcular a Transformada Discreta de Fourier, que é aplicado em muitas áreas como por exemplo processamento digital de sinais,
    E Muito dos efeitos que este algoritmo produz são de tal forma críticos, que a sua implementação é feita muitas das vezes em GPU para obter o máximo de eficiencia possivel

<!-- press next -->

Os objetivos principais deste projeto são estudar implementações de Transformadas rápidas de Fourier no GPU ao produzir implementações em GLSL comparáveis com a framework de FFT em CUDA (cuFFT), e ainda aplicar optimizações para implementações especializadas desta transformada em aplicações.

Para alcançar estes objetivos esta dissertação está estruturada em 3 etapas principais.

## Etapas

Numa primeira fase da dissertação é estudado o estado de arte dos algoritmos que já existem enquadrados em Transformadas Rápidas de Fourier, tanto os sequenciais como paralelos, e vai também englobar propriedades e especificações de computação que otimizam as implementações.

Este estágio vai ser essencial para o segundo capítulo de estudo, em que vai ser feita a comparação detalhada de implementações de FFT's
no GPU entre CUDA pela framework popular cuda FFT e GLSL.

Finalmente isto vai culminar na Integração e optimização das implementações no contexto de aplicações, neste caso, que requerem performance em tempo real como por exemplo renderização de ondas de oceanos.

<!-- Nota sobre a expansão do tema -->
- Apesar desta dissertação estar focada em transformadas rapidas de fourier, o conhecimento pode-se expandir para qualquer implementação de algoritmos paralelos e ainda demonstra de que forma se pode aplicar estas implementações em Computação Gráfica.

<!-- ## Resultados esperados

- No final espera-se que na comparação entre cuda FFT e glsl, que cuda FFT tenha ainda uma performance superior por estar especializada para o hardware em que corre, enquanto que que a nossa implementação de GLSL vai ter suporte para maior parte dos GPU's fornecendo na mesma uma performance acima da média. -->
<!-- Para além disto, espera-se que no caso de estudo das ondas de Tensendorf, ter apenas a API gráfica a lidar com os compute kernels tenha  -->


## Resumo de algum trabalho

Esta parte vai servir como breve introdução dos conceitos em que esta dissertação vai desenvolver


### Discrete Fourier Transform

- Uma transformada de Fourier discreta é um método que converte uma sequência do domínio referido como tempo
para o domínio de frequência. Como por exemplo o processo de disseminação de um sinal complexo em sinais básicos sinusoidais

- É representado por um somatório de multiplicações de números no domínio dos complexos por isso é comum
encontrar implementações na sua forma matricial

- Mas apesar da utilidade deste método, ele tem uma complexidade de operações O(n^2) o que não escala muito bem em performance
para sequencias de input maiores, e há a necessidade de adotar algoritmos mais eficientes

### Fast Fourier Transforms

E é aí que as trasnformadas rápidas de Fourier brilham, são uma familia de algoritmos, em que a implementação mais popular
é a de Cooley-Tukey para sequencias de input de tamanho potencia de 2, e há ainda variantes da fatorização do input que aproveitam
a eficáfia deste algoritmo a a aplicam para outros tamanhos de inputs para além das potencias de 2.
Além disso estes algoritmos também costumam ser altamente paralelizáveis

Algumas das variantes incluem:

- A Decimation-in-Time, a mais popular
- Decimation-in-Frequency
- DIT recursiva
- O PFA (Prime Factor Algorithm)
E também os algoritmos de Rader e Bluestein

## Comparação de resultados entre estes algoritmos só

Nessa tabela pode-se observar a diferença notável entre o cálculo da transformada discreta e algoritmos da trasformada rápida de Fourier

São tudo comparações de algoritmos feitos em C++ compiladas com O2, desenvolviods especificamente para esta dissertação.

Nota-se a grande diferença de performance, o que no caso da DFT não é só influenciado pela complexidade de tempo mas também pela complexidade de espaço visto que é feita por um produto matricial.

Isto dá uma noção do quanto se pode aprimorar no calculo de DFT com algoritmos mais sofisticados

## Proximos passos

Finalmente, na calendarização mantém-se 4 etapas planeadas tal como no plano de trabalho proposto,
excluindo a escrita da dissertação e pré-dissertação, que são

- Investigação de Transformadas Rápidas de Fourier
- Estudo da framework cuda FFT
- Análise de kernels de CUDA e GLSL
- Investigação de FFT especializadas  aplicações

Adicionalmente, já há alguma antecipação das etapas seguintes
