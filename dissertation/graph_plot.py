import matplotlib.pyplot as plt
import re

# plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

MARKERS = ['o', 'v', '^', '>', '<', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

to_snakecase = lambda s : re.sub(r'((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))', r'_\1', s).lower()

def ms_graph(title, sizes, data, filename=''):
    default_sizes_ticks = range(len(sizes))
    plt.xticks(default_sizes_ticks, sizes)
    for i,(lbl,ms) in enumerate(data):
        plt.plot(default_sizes_ticks, ms, label=lbl, marker=MARKERS[i])
    plt.xlabel('FFT size')
    plt.ylabel('ms')
    plt.margins(0.1, 0.2)
    plt.title(title)
    plt.legend()
    # Save to file
    if filename != '':
        plt.savefig(f'{filename}.png')
    plt.show()


# NOTE: Just a test
# ms_graph("Frame time",
#     sizes=[128, 256, 512],
#     data=[
#         ("fft", [1.312, 1.411, 1.577]),
#         ("fft_mb", [1.361, 1.501, 1.604]),
#         ("fft_mb_gpuonly", [0.743, 0.752, 0.742]),
#         ("fft_stockham", [0.687, 0.719, 0.789]),
#     ]
# )

#################################

# ms_graph("cuFFT vs GLSL Radix-2",
#     sizes=[128, 256, 512, 1024],
#     data=[
#         ("cuFFT", [0.0359, 0.0494, 0.1335, 0.5609]),
#         ("GLSL Radix-2 Cooley-Tukey", [0.073, 0.257, 1.032, 2.646]),
#         ("GLSL Radix-2 Stockham",[0.049, 0.135, 0.545, 2.341]),
#     ]
#     , filename='cufft_vs_glsl_radix2'
# )

# ms_graph("cuFFT vs GLSL Radix-4",
#     sizes=[256, 1024],
#     data=[
#         ("cuFFT", [0.0494, 0.5609]),
#         ("GLSL Radix-2 Stockham",[0.135, 2.341]),
#         ("GLSL Radix-4 Stockham",[0.087, 1.363]),
#     ]
#     , filename='cufft_vs_glsl_radix4'
# )


# ms_graph("CUDA Radix-2 vs GLSL Radix-2",
#     sizes=[128, 256, 512, 1024],
#     data=[
#         # CUDA
#         ("CUDA Radix-2 Cooley-Tukey", [0.057, 0.220,  0.944, 3.729]),
#         ("CUDA Radix-2 Stockham",[0.054, 0.209, 1.047, 4.813]),
#         # GLSL
#         ("GLSL Radix-2 Cooley-Tukey", [0.073, 0.257, 1.032, 2.646]),
#         ("GLSL Radix-2 Stockham",[0.049, 0.135, 0.545, 2.341]),
#     ]
#     , filename='cuda_vs_glsl_radix2'
# )

# ms_graph("CUDA vs GLSL Radix-4",
#     sizes=[256, 1024],
#     data=[
#         ("CUDA Radix-2 Stockham", [0.209, 4.813]),
#         ("CUDA Radix-4 Stockham", [0.123, 1.993]),
#         ("GLSL Radix-2 Stockham",[0.135, 2.341]),
#         ("GLSL Radix-4 Stockham",[0.087, 1.363]),
        
#     ]
#     , filename='cuda_vs_glsl_radix4'
# )

# ms_graph("Frame time Stage per pass vs unique pass FFT",
#     sizes=[128, 256, 512, 1024],
#     data=[
#         ("Cooley-Tukey stage per pass", [1.361, 1.501, 1.604, 1.806]),
#         ("Cooley-Tukey unique pass", [0.743, 0.752, 0.786, 0.79]),
#     ]
#     , filename='glsl_stage_pass_vs_unique_pass'
# )

ms_graph("Radix-2 Stockham with multiple butterflies",
    sizes=[128, 256, 512, 1024],
    data=[
        ("1 Butterfly", [0.049, 0.135, 0.545, 2.341]),
        ("2 Butterfly", [0.067, 0.234, 0.934, 4.104]),
        ("4 Butterfly", [0.092, 0.298, 3.047, 14.934]),
    ]
    , filename='glsl_multiple_butterflies'
)