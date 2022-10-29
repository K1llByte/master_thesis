import matplotlib.pyplot as plt
# plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

def ms_graph(title, sizes, data):
    default_sizes_ticks = range(len(sizes))
    plt.xticks(default_sizes_ticks, sizes)
    for lbl,ms in data:
        plt.plot(default_sizes_ticks, ms, label=lbl)
    plt.xlabel('FFT size')
    plt.ylabel('ms')
    plt.margins(0, 0.2)
    plt.title(title)
    plt.legend()
    plt.show()

    # Save to file
    # title.to_snake_case e tal
    # plt.savefig('my_plot.png')


ms_graph("Frame time",
    sizes=[128, 256, 512],
    data=[
        ("fft", [1.312, 1.411, 1.577]),
        ("fft_mb", [1.361, 1.501, 1.604]),
        ("fft_mb_gpuonly", [0.743, 0.752, 0.742]),
        ("fft_stockham", [0.687, 0.719, 0.789]),
    ]
)