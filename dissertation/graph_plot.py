import matplotlib.pyplot as plt
import re

# plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

MARKERS = ['o', 'v', '^', '>', '<', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

to_snakecase = lambda s : re.sub(r'((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))', r'_\1', s).lower()

def ms_graph(title, sizes, data, filename='', lstyles=[], colors=[], ylim=None):
    assert(colors == [] or len(colors) == len(data))
    default_sizes_ticks = range(len(sizes))
    plt.xticks(default_sizes_ticks, sizes)
    for i,(lbl,ms) in enumerate(data):
        style = '-' if lstyles == [] else lstyles[i]
        if colors == []:
            plt.plot(default_sizes_ticks, ms, label=lbl, linestyles=style, marker=MARKERS[i])
        else:
            plt.plot(default_sizes_ticks, ms, label=lbl, color=colors[i], linestyle=style, marker=MARKERS[i])
    plt.xlabel('FFT size')
    plt.ylabel('ms')
    if ylim != None:
        # plt.set_ylim(0,ylim)
        plt.ylim(0,ylim)
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

ms_graph("GLSL and cuFFT Forward 2D FFT benchmarks",
    sizes=[128, 256, 512, 1024],
    data=[
        ("cuFFT", [0.0359, 0.0494, 0.1335, 0.5609]),
        ("GLSL Radix-2 Cooley-Tukey", [0.073, 0.257, 1.032, 2.646]),
        ("GLSL Radix-2 Stockham",[0.049, 0.135, 0.545, 2.341]),
        ("GLSL Radix-4 Stockham",[0.042, 0.087, 0.389, 1.363]),
    ]
    , filename='cufft_glsl_benchmarks'
    , colors=['tab:pink', 'tab:red', 'tab:green', 'tab:blue']
)

# ms_graph("cuFFT vs GLSL Radix-4",
#     sizes=[256, 1024],
#     data=[
#         ("cuFFT", [0.0359, 0.0494, 0.1335, 0.5609]),
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

# ms_graph("Radix-2 Stockham with multiple butterflies",
#     sizes=[128, 256, 512, 1024],
#     data=[
#         ("1 Butterfly", [0.049, 0.135, 0.545, 2.341]),
#         ("2 Butterfly", [0.067, 0.234, 0.934, 4.104]),
#         ("4 Butterfly", [0.092, 0.298, 3.047, 14.934]),
#     ]
#     , filename='glsl_multiple_butterflies'
# )



# ms_graph("Stockham for power of 2 sizes",
#     sizes=[128, 256, 512, 1024],
#     data=[
        
#         ("GLSL Radix-2 Stockham",[0.049, 0.135, 0.545, 2.341]),
#         ("GLSL Radix-4 Stockham",[0.042, 0.087, 0.389, 1.363]),
#     ]
#     , filename='glsl_stockham'
# )


ms_graph("GLSL and cuFFT 2 forward 2D FFT benchmarks",
    sizes=[128, 256, 512, 1024],
    data=[
        ("cuFFT double", [0.036, 0.098, 0.230, 1.031]),
        ("cuFFT single", [0.0359, 0.0494, 0.1335, 0.5609]),
        ("GLSL Radix-4 Stockham double", [0.044, 0.132, 0.703, 2.702]),
        ("GLSL Radix-4 Stockham single", [0.042, 0.087, 0.389, 1.363]),
    ]
    , filename='cufft_glsl_multiple_fft_benchmarks'
    , colors=['tab:pink','tab:pink','tab:blue','tab:blue']
    , lstyles=['-', '--','-', '--']
)

# ms_graph("GLSL and cuFFT 2 forward 2D FFT benchmarks",
#     sizes=[128, 256, 512, 1024],
#     data=[
#         ("cuFFT", [0.037/2, 0.098/2, 0.230/2, 1.031/2]),
#         ("GLSL Radix-4 Stockham", [0.044/2, 0.132/2, 0.703/2, 2.702/2]),
#     ]
#     , filename='cufft_glsl_multiple_fft_benchmarks_per_fft'
# )

# def diff(la, lb, label=""):
#     print([(a-b) / a * 100 for a,b in zip(la,lb)])


# diff(
#     [0.035, 0.049, 0.133, 0.560],
#     [0.019, 0.049, 0.115, 0.516],
#     label="cuFFT"
# )

# diff(
#     [0.042, 0.087, 0.389, 1.363],
#     [0.022, 0.066, 0.352, 1.351],
#     label="GLSL"
# )

################ Unique vs Stage ################

ms_graph("CPU time of forward 2D FFT benchmarks in GLSL",
    sizes=[128, 256, 512, 1024],
    data=[
        ("Unique pass", [0.034 + 0.026, 0.030 + 0.022, 0.029 + 0.020, 0.039 + 0.021]),
        ("Stage per pass", [0.194 + 0.189, 0.210 + 0.192, 0.234 + 0.210, 0.255 + 0.245]),
    ]
    , filename='glsl_stage_pass_vs_unique_pass_cpu'
    , ylim = 4
    , colors=['tab:red', 'tab:orange']
)

ms_graph("GPU time of forward 2D FFT benchmarks in GLSL",
    sizes=[128, 256, 512, 1024],
    data=[
        ("Unique pass", [0.026+0.024, 0.078+0.072, 0.305+0.293, 1.390+1.308]),
        ("Stage per pass", [0.055+0.055, 0.100+0.099, 0.500+0.473, 1.988+1.911]),
    ]
    , filename='glsl_stage_pass_vs_unique_pass_gpu'
    , colors=['tab:red', 'tab:orange']
)