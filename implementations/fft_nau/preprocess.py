from PIL import Image

FFT_SIZE = 256

def preprocess(input_img):
    output_img = input_img.copy()
    # for w in range(1):
    for w in range(input_img.width):
        # for h in range(1):
        for h in range(input_img.height):
            # Per pixel operations
            # 1. Relative luminance to complex
            (r,g,b) = input_img.getpixel((w,h))
            lum = 0.212671*r + 0.715160*g + 0.072169*b
            output_img.putpixel((w,h), (round(lum), 0, 0))

    return output_img

def compute_twiddle_texture(size):
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    img = Image.new("F", size)
    for w in range(img.width):
        for h in range(img.height):
            
            img.putpixel((w,h), 255.)
    return img
    
if __name__ == '__main__':
    # with Image.open("doggy.jpg") as input_img:
    #     output_img = preprocess(input_img)
    #     print(f"{output_img.getpixel((0,0))} ", end='')
    #     output_img.show()
    #     output_img.save("preprocessed_doggy.jpg")
    size = (FFT_SIZE, FFT_SIZE)
    img = compute_twiddle_texture(size)
    img.show()