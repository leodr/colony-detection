import cv2
import numpy as np


class CircleDetectionHyperparameters:
    def __init__(
        self,
        blur_kernel_size=15,
        blur_sigmax=0,
        outer_circle_param1=150,
        outer_circle_param2=10,
        colony_minDist=10,
        colony_param1=50,
        colony_param2=30,
        colony_minRadius=3,
        colony_maxRadius=50,
    ):
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigmax = blur_sigmax
        self.outer_circle_param1 = outer_circle_param1
        self.outer_circle_param2 = outer_circle_param2
        self.colony_minDist = colony_minDist
        self.colony_param1 = colony_param1
        self.colony_param2 = colony_param2
        self.colony_minRadius = colony_minRadius
        self.colony_maxRadius = colony_maxRadius


class CircleDetection:
    """
    Algorithm for detecting colonies in an image, using hough circle detection.
    """

    def __init__(self, hyperparameters=CircleDetectionHyperparameters()):
        self.hyperparameters = hyperparameters

    def __create_circles(self, image):
        height, width, channels = image.shape

        # Convert the image to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use Gaussian blur to reduce noise and improve circle detection
        blurred = cv2.GaussianBlur(
            gray,
            (
                self.hyperparameters.blur_kernel_size,
                self.hyperparameters.blur_kernel_size,
            ),
            self.hyperparameters.blur_sigmax,
        )

        # Detect the petri dish
        outer_circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=6,
            minDist=30,
            param1=self.hyperparameters.outer_circle_param1,
            param2=self.hyperparameters.outer_circle_param2,
            # Circle should be large, but not larger than the image.
            minRadius=round(max(height, width) / 3),
            maxRadius=round(max(height, width) / 2),
        )

        if outer_circles is not None:
            # Convert circles to expected format
            outer_circles = np.uint16(np.around(outer_circles))

            # Create a black mask
            mask = np.zeros_like(gray)

            # Get the (x, y) coordinate and the radius of the circle
            x, y, r = outer_circles[0][0]

            # Draw the circle in the mask
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

            # Bitwise the image with mask
            result = cv2.bitwise_and(blurred, mask)

            blurred = result

        # Detect circles in the image
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.hyperparameters.colony_minDist,
            param1=self.hyperparameters.colony_param1,
            param2=self.hyperparameters.colony_param2,
            minRadius=self.hyperparameters.colony_minRadius,
            maxRadius=self.hyperparameters.colony_maxRadius,
        )

        return circles

    def create_mask(self, image: np.ndarray):
        """
        Create a mask.

        Parameters:
        image (np.ndarray): The RGB image to create a mask from.

        Returns:
        np.ndarray: The mask, where 255 is a colony and 0 is not a colony.
        """
        circles = self.__create_circles(image)

        # Create a black mask
        mask = np.zeros_like(image)

        if circles is not None:
            # Convert circles to expected format
            circles = np.uint16(np.around(circles))

            # Draw the circles in the mask
            for circle in circles[0]:
                x, y, r = circle
                cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

        return mask

    def count_colonies(self, image):
        """
        Count the number of colonies in an image.

        Parameters:
        image (np.ndarray): The RGB image to count colonies in.

        Returns:
        int: The number of colonies in the image.
        """
        circles = self.__create_circles(image)

        if circles is None:
            return 0

        return len(circles[0])
