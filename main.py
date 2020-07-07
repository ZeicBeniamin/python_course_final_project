import cv2
import glob
from functools import partial
import numpy as np
import matplotlib.image as mpimg
from plot import Plot

gaus_ker = 13

diam = 9
sig_col = 5
sig_sp = 31

sob_ker = 5
mag_ker = 3


def abs_sobel_thresh(img, thresh=(0, 255), orient='x', sobel_kernel=sob_ker):
    '''
    Calculate directional gradient.
    '''

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if orient == 'x':
        sobel_dir = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel_dir = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sobel_abs = np.absolute(sobel_dir)

    max_sobel = np.max(sobel_abs)
    scaled = np.uint8(255 * (sobel_abs / max_sobel))

    grad_binary = np.zeros_like(scaled)
    map = (scaled >= thresh[0]) & (scaled < thresh[1])
    grad_binary[(scaled >= thresh[0]) & (scaled < thresh[1])] = 1

    return grad_binary


def mag_thresh(image, mag_thresh=(0, 255), sobel_kernel=mag_ker):
    '''Calculate gradient magnitude'''

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    magnitude = np.sqrt((sobel_x ** 2) + (sobel_y ** 2))

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled = np.uint8(magnitude * 255 / np.max(magnitude))
    # 5) Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(scaled)
    mag_binary[(scaled >= mag_thresh[0]) & (scaled < mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image

    return mag_binary


def dir_threshold(image, thresh=(0, np.pi / 2), sobel_kernel=3):
    '''Calculate gradient direction'''

    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    absx = np.absolute(sobelx)
    absy = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(absy, absx)
    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(direction)
    dir_binary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image

    return dir_binary


def hls_select(imag, thresh=(0, 1), channel='s'):
    hls = cv2.cvtColor(imag, cv2.COLOR_RGB2HLS)

    if channel == 'h':
        ch = 0
    elif channel == 'l':
        ch = 1
    else:
        ch = 2

    layer = hls[:, :, ch]

    # 3) Return a binary image of threshold result
    mask = np.zeros_like(layer)

    mask[(layer >= thresh[0]) & (layer <= thresh[1])] = 1

    return mask


def gamma_correction(img_original, thresholds=(0, 10)):
    # The image provided is of type float32 with values ranging from 0 to 1, so
    # we need to rescale it by 255 in order to convert it to int without loosing
    # essential information

    img_original = cv2.convertScaleAbs(img_original * 255)

    # Gamma in range 0.01 and 25.0

    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, thresholds[0]) * 255.0, 0, 255)

    res = cv2.LUT(img_original, lookUpTable)

    x.gamma = res

    # Convert back to float32
    res = res.astype(np.float32)
    res = res / 255

    return res


def lab_select(image, thresh=(0, 255), channel='l'):
    cie_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    if channel == 'l':
        ch = 0
        channel = "CIE L"
    elif channel == 'a':
        ch = 1
        channel = "CIE A"
    elif channel == 'b':
        ch = 2
        channel = "CIE B"

    layer = cie_lab[:, :, ch]

    thresh0 = thresh[0]
    thresh1 = thresh[1]

    # 3) Return a binary image of threshold result
    mask = np.zeros_like(layer)

    mask[(layer >= thresh0) & (layer <= thresh1)] = 1

    return mask


count = 1

# Store the slider name, minimum, maximum and initial values in a list
sliders = [
    ['abs', -1, 255, 0, 0.1],
    ['', -1, 255, 255, 0.1],
    ['mag', 0, 255, 0, 1],
    ['', 0, 255, 255, 1],
    ['dir', 0, 255, 0, 1],
    ['', 0, 255, 255, 1],
    [],
    [],
    ['H', 0, 100, 0, 1],
    ['', 0, 100, 100, 1],
    ['L', 0, 1.5, 0, 0.01],
    ['', 0, 1.5, 1.5, 0.01],
    ['S', 0, 1.5, 0, 0.01],
    ['', 0, 1.5, 1.5, 0.01],

    ['gam', 0, 13, 1, 0.01],
    ['n', 0, 13, 0, 0.01],

    ['H', 0, 100, 0, 1],
    ['h2', 0, 100, 100, 1],
    ['L', -127, 10, 0, 0.01],
    ['l2', -127, 10, 1.5, 0.01],
    ['S', -101, 101, 0, 1],
    ['s2', -101, 101, 1.5, 1],
]


# Define an array with all the necesarry arguments for each slider
# `[name, min_val, max_val, valinit, step]`
def press(event):
    global count
    if event.key == 'x':
        print("new image")
        x.set_image(get_next_img_blurred(1))
    elif event.key == 'z':
        x.set_image(get_next_img_blurred(5))
    elif event.key == 'c':
        x.set_image(get_next_img_blurred(10))
    elif event.key == 'v':
        x.set_image(get_next_img_blurred(23 * 23))


def raw_image():
    return x.get_image()


cap = cv2.VideoCapture(
    '/home/phantomcoder/Desktop/self_driving_car_udacity/CarND-Advanced-Lane-Lines/project_video.mp4')


def get_next_img_blurred(frames_number=1):
    global images
    global count

    fname = next(images)

    fname = f"/home/phantomcoder/Desktop/self_driving_car_udacity/CarND-Advanced-Lane-Lines/problem_frames/{count}.png"
    img = mpimg.imread(fname)

    # For use with video
    # while frames_number:
    #     ret, img = cap.read()
    #     frames_number -= 1

    ''' TEMPORARY ~~~~~~~~~~~!!!!!!!!!!!!!!!!! DELETE AS SOON AS POSSIBLE'''
    # img = cv2.imread('/home/phantomcoder/Desktop/self_driving_car_udacity/CarND-Advanced-Lane-Lines/test_images/7.jpg')

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # #
    # # img = img.astype(np.float32)
    # # img = img / 255

    img = cv2.bilateralFilter(img, d=diam, sigmaColor=sig_col, sigmaSpace=sig_sp)

    count += frames_number

    return img


def combined_image():
    image = raw_image()

    sobel_abs = abs_sobel_thresh(raw_image(), (x.get_slider_value(0), x.get_slider_value(1)))
    grad_mag = mag_thresh(image, (x.get_slider_value(2), x.get_slider_value(3)))

    h_white = hls_select(image, (x.get_slider_value(8), x.get_slider_value(9)), 'h')
    h_yellow = hls_select(image, (x.get_slider_value(16), x.get_slider_value(17)), 'h')

    l_chan = hls_select(image, (x.get_slider_value(10), x.get_slider_value(11)), 'l')
    s_chan = hls_select(image, (x.get_slider_value(12), x.get_slider_value(13)), 's')

    cie_l = lab_select(image, (x.get_slider_value(16), x.get_slider_value(17)), 'l')
    cie_b = lab_select(image, (x.get_slider_value(20), x.get_slider_value(21)), 'b')

    combined = np.zeros_like(s_chan)

    combined[(sobel_abs == 1) &
             (grad_mag == 1) &
             (
                     (h_white == 1) |
                     (h_yellow == 1) |
                     (l_chan == 1) |
                     (s_chan == 1)
             )] = 1

    # combined[(cie_l == 1) |
    #          (cie_b == 1)
    #          ] = 1

    return combined


images = glob.glob("./problem_frames/*.png")
images = iter(images)

x = Plot(3, 4)
x.set_plot()
x.set_bars()
x.set_sliders(sliders)

# x.set_result_function(7, combined_image)
x.set_callback(0, abs_sobel_thresh)
x.set_callback(1, mag_thresh)
x.set_callback(2, partial(abs_sobel_thresh, orient='y'))
x.set_callback(3, raw_image)
x.set_callback(4, partial(hls_select, channel='h'))
x.set_callback(5, partial(hls_select, channel='l'))
x.set_callback(6, partial(hls_select, channel='s'))

# Remove "image modifying" subplot
# gamma_subplot_idx = 7
# x.set_one_to_many(gamma_subplot_idx, gamma_correction, [8, 9, 10])

x.set_callback(8, partial(lab_select, channel='l'))
x.set_callback(9, partial(lab_select, channel='a'))
x.set_callback(10, partial(lab_select, channel='b'))
x.set_result_function(11, combined_image)

x.set_keyboard_callback(press)

x.initialize_images(get_next_img_blurred())

x.show_plot()
