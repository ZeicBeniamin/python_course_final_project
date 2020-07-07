import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

'''
    Plot creates a matplotlib plot, containing multiple subplots and a slider for each subplot.
    
    The subplots can be:
     - normal subplots 
     - "result" subplots
    - "image modifying" subplots
    
    "Normal" subplots have two sliders attached to them, that adjust the lower and upper 
    threshold of an image filter. 
    Result subplots, on the other hand, combine the resulting (i.e. filtered) images of 
    multiple normal subplots and display the new result.
    "Image modifying" subplots take the original image, apply a filter over it and then
    they send the modified image (filtered image) to other "normal" subplots, that apply
    other filters on it. 
    
    In the current version of the project, only the bottom left (row 3, column 4) subplot 
    is a "result" subplot. The subplot (row 2, column 4) was an "image modifying" subplot
    but it has now been removed
'''


class Plot:

    def __init__(self, rows=2, cols=3, figsize=(15, 10)):
        ''' Initialize all the parameters of this plot'''
        self.dx = 0
        self.dy = 0
        self.height = 0
        self.width = 0
        self.gap = 0

        self.fig = None
        self.axes = []
        self.ax_imgs = []
        self.bars = []
        self.sliders = []
        self.axcolor = None
        self.image = None
        self.result_idx = None
        self.gamma = None

        self.related_subpl = None

        self.press_func = None
        self.result_func = None
        self.callbacks = []

        self.rows = rows
        self.cols = cols
        self.subplots = self.rows * self.cols
        self.figsize = figsize

        # Create an empty list of callbacks, so that we can use it as an indexed array
        # of predefined size
        self.callbacks = [None for i in range(self.rows * self.cols)]

        # Initialize a list of related subplots for each subplot
        self.related_subpl = [[] for i in range(self.rows * self.cols)]

    def set_plot(self, top=1, right=0.97, bottom=0.1, left=0.03, wspace=0.2):
        '''
        .   @brief Sets up the subplots and the plot's margins

        :param top: Position of the top margin of the plot with respect to the window
        :param right: Position of the right margin of the figure with respect to the window
        :param bottom: Position of the bottom margin of the figure with respect to the window
        :param left: Position of the left margin of the figure with respect to the window
        :param wspace: Width of free space between subplots

        '''
        self.fig, ax = plt.subplots(self.rows, self.cols, figsize=self.figsize)

        for row in ax:
            for ax in row:
                self.axes.append(ax)

        plt.subplots_adjust(top=top, right=right, bottom=bottom, left=left, wspace=wspace)

    def set_bars(self, dx=0.01, dy=0.05, height=0.03, width=0.8, gap=0.06, axcolor='lightgoldenrodyellow'):
        '''
        Sets the position of the axes (bars) that will be used as sliders

        :param dx: Horizontal distance between neighbouring axes (parameter not used yet)
        :param dy: Another parameter for vertical distance between bars (the first one is `gap`)
        :param height: Height of axes (bars)
        :param width: Width of axes (bars)
        :param gap: Vertical distance between axes (bars) corresponding to the same subplot
        :param axcolor: Color of the axes(bars)

        :return: None
        '''
        self.dx = dx * (1 / self.cols)
        self.dy = dy * (1 / self.rows)
        self.height = height * (1 / self.rows)
        self.width = width * (1 / self.cols)
        self.gap = gap * (1 / self.rows)
        self.axcolor = axcolor

        # Calculate the percentage of horizontal (`nc`) and vertical (`nr`) space that each subplot
        # occupies within the plot
        nc = 1 / self.cols
        nr = 1 / self.rows

        # Generate two axes for each subplot
        for i in range(self.rows - 1, -1, -1):
            for j in range(self.cols):
                # The position of each subplot is calculated using the `calculate_ax_pos` method
                self.bars.append(plt.axes(self.calculate_ax_pos(j, i, True), facecolor=axcolor))
                self.bars.append(plt.axes(self.calculate_ax_pos(j, i), facecolor=axcolor))

    def set_sliders(self, sliders_prop):
        '''
        Sets up the sliders (together with their properties) for each ax (bar).

        :param sliders_prop: List of lists, containing the properties of the sliders.
            NOTE: If a sublist is empty, no slider will be attached to the corresponding ax.
            The slider will still be visible in the UI but it wil have no functionality attached
            A sublist of `sliders_prop` must contain the parameters in the following order:
            slider name, minimum slider value, maximum slider value, initial slider value,
            step of the slider (amount of a single incrementation).
        :return: None

        '''

        # Take each slider's properties, together with its index
        for index, props in enumerate(sliders_prop):

            # Check if there is any data in the properties list
            # If so, then proceed to initializing the slider
            if len(props) > 0:

                name = props[0]
                min = props[1]
                max = props[2]
                init = props[3]
                step = props[4]

                self.sliders.append(Slider(self.bars[index], name, min, max, valinit=init, valstep=step))
            # Otherwise, append a `None` object to the list. That helps correctly indexing the sliders,
            # even if some sliders are not
            else:

                self.sliders.append(None)

    def set_callback(self, subplot_idx, function):
        '''
        Attaches a callback function to the sliders corresponding to a certain subplot

        A custom callback function is first generated from a generic function

        :param subplot_idx: Zero-based index of the subplot to attach the callback to. The indexes begin
            with the upper left corner and continue along the columns (such that the last subplot is
            located in the lower right corner.
        :param function: Function to be passed as callback. The function must accept only three positional
            arguments (it can have more than three arguments, but they must have a default value). The order of
            the arguments is as follows: image, threshold minimum value, threshold maximum value
        :return: None
        '''

        # Create a personalized callback function for the subplot with index subplot_idx
        callback_function = self.generic_callback(subplot_idx, function)
        # Connect the sliders below this subplot to the callback method created

        for slider in self.sliders[subplot_idx * 2: subplot_idx * 2 + 2]:
            #
            if slider != None:
                slider.on_changed(callback_function)
                self.callbacks[subplot_idx] = callback_function
            # This is for "result subplots"
            # "Result subplots" only combine different images, they don't adjust their properties
            # so they have no sliders, but they must have a callback function that is called
            # when a subplot tied to the "result subplot" modifies its state
            else:
                self.callbacks[subplot_idx] = callback_function

    def set_one_to_many(self, subplot_idx, function, related_subpl=None):
        '''
        Adds a callback to a subplot that has a "one to many" relationship with other subplots.
        The related subplots' callbacks will be called whenever changes occur in the current subplot

        :param subplot_idx: Index of subplot
        :param function: Function to be called in callback
        :param related_subpl: List of subplots that must be modified when modifying the current subplot
        :return: None
        '''

        self.set_callback(subplot_idx, function)

        # Add all the related subplots to a list, such that in the future we will know what callback function to call.
        for subpl in related_subpl:
            self.related_subpl[subplot_idx].append(subpl)

    def set_keyboard_callback(self, press_function):
        '''
        Set function that will handle keyboard events
        :param press_function: Function that will handle events
        :return: None
        '''
        self.fig.canvas.mpl_connect('key_press_event', press_function)

    def set_image(self, image):
        '''
        Changes the image with which the class will work

        :param image: cv2 Image that will be processed
        :return: None
        '''
        self.image = image
        self.update_all_subplots()

    def get_image(self):
        '''
        Returns the image currently stored in the object
        :return: Image
        '''
        return self.image

    def generic_callback(self, subplot_idx, function):
        '''
        Generate different callback methods for sliders

        If this method would not have been used, a different callback should have been created for
        each slider (or each slider groups). Using this method lets us change the behaviour of
        `wrapper_generic_callback` without writing multiple callback functions.

        :param subplot_idx: Index of subplot for which the callback will be created
        :param function: Function to be passed to callback (function must return an image)
            NOTE: If there are sliders linked to the subplot, the function passed as argument must
            accept 3 arguments, in this order: image, minimum threshold, maximum threshold
            Otherwise, it should expect no arguments.
        :return: None
        '''

        def wrapper_generic_callback(val):

            # Explanation (attempt) of the messy code below:
            # Callbacks set on sliders only update their specific subplot. If we want a "result subplot",
            # that subplot must be updated for every change in the related subplots (that's why we perform the last
            # check " if self.result_fun " ). If any result function is set, we call it, without concerning
            # which subplot was modified (since the result function takes directly the slider values, it can process
            # the image without needing to know which function called it).

            # If we've got a callback for a subplot that has sliders, get the values of the sliders for this subplot
            # and then pass them to the function
            try:
                # Get values from the sliders corresponding to this subplot number
                slider_vals = [slider.val for slider in self.sliders[2 * subplot_idx: 2 * subplot_idx + 2]]
                # Pass those values to the function, together with an image
                image = function(self.image, (slider_vals[0], slider_vals[1]))
            # Otherwise, if the subplot has no sliders (that means it is a "result subplot, call the function
            # by passing only the image as an argument
            # This is not necessary when editing a single image, but when changing between images,
            # all the callbacks are called and all the images are updated.

            # In this case, trying to get the slider values for a result subplot would raise an error
            except:
                image = function()

            # The image resulted after the code above is executed is either the image for a normal subplot (if the
            # subplot has sliders), either the image of a result function (if the subplot doesn't have sliders).
            # The latter case only happens when changing the image using `set_image`

            # Update the image for the current subplot
            self.update_image(subplot_idx, image)

            # Call the result function every time a new event occurs, without caring if the event occurs in a
            # subplot related to the result subplot
            # if self.result_idx == subplot_idx:
            if self.result_func:
                self.update_image(self.result_idx, self.result_func())

            # If teh current subplot has other related subplots
            if len(self.related_subpl[subplot_idx]) > 0:
                # Call the callback of each subplot, such that those subplots will update their data
                for subpl in self.related_subpl[subplot_idx]:
                    self.callbacks[subpl](1)

        return wrapper_generic_callback

    def update_image(self, subplot_idx, image):
        '''
        Updates subplot to display the `image` passed as argument.

        :param subplot_idx: Index of the subplot we want to modify
        :param image: Image to be displayed in the subplot

        :return: None
        '''
        self.ax_imgs[subplot_idx].set_data(image)
        # self.ax_imgs[subplot_idx].set_cmap('binary')
        self.fig.canvas.draw_idle()

    def set_result_function(self, subplot_idx, result_func):
        '''

        :param subplot_idx:
        :param result_func:
        :return:
        '''
        self.result_idx = subplot_idx
        self.result_func = result_func

    def get_slider_value(self, index):

        return self.sliders[index].val

    def initialize_images(self, image):
        '''
        Initializes all subplots with the same image, received as parameter

        The image should be a color image in order not to have problems.
        This step is required because, for some reason, matplotlib does not display future images correctly
        if the first image displayed was a binary image (containing only ones and zeros)
        :param image: Color image. If a one channel image is sent, errors will be raised in
            other methods.
        :return: None
        '''

        self.image = image
        for ax in self.axes:
            img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            self.ax_imgs.append(ax.imshow(img, cmap='gray'))

    def update_all_subplots(self):
        for callback in self.callbacks:
            if callback != None:
                callback(1)

    def calculate_ax_pos(self, c, r, gap=False):
        '''
        Auxiliary method that calculates the position of the axes (bars) within the current plot

        NOTE: Row and column numbers are indexed like in a cartesian coordinate plane having the origin
        in the bottom left of the screen (i.e. row number increases upwards and column number increases rightwards)
        :param c: Column in which the current subplot lies
        :param r: Row in which the current subplot lies
        :param gap: Whether or not to add more distance on the vertical axis

        :return: None
        '''
        nc = 1 / self.cols
        nr = 1 / self.rows

        if gap:
            return [c * nc + (1 / 2) * nc - self.width / 2, r * nr + self.dy + self.gap, self.width, self.height]
        else:
            return [c * nc + (1 / 2) * nc - self.width / 2, r * nr + self.dy, self.width, self.height]

    @classmethod
    def show_plot(cls):
        '''
        Draw the GUI of the current plot
        :return: None
        '''
        plt.show()
