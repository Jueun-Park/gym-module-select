import cv2
import numpy as np

RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)


class LaneDetector:
    def __init__(self):
        self.angle = 0
        self.previous_angles = []

    def __insert_previous_angle(self):
        max_size = 10
        if len(self.previous_angles) >= max_size:
            del self.previous_angles[0]
        self.previous_angles.append(self.angle)

    def detect_lane(self, image: np.ndarray):
        self.original_image_array = image.copy()
        self.image_array = image.copy()
        self.right = []
        self.left = []
        self.done = True

        self._blur_image()
        self._get_grayscale_image()
        self._detect_edges()
        self._get_roi_image()
        self._hough_line()
        if self.hough_lines is None:
           return False, self.angle  # not done
        self._get_lane_candidates()
        self._intersection_of_lanes()
        self._get_angle_error()

        add_term = 0.1  # TODO: consider previous angles
        if not self.right or not self.left:
            self.angle = np.mean(self.previous_angles)
            if not self.right:
                self.angle += add_term
            if not self.left:
                self.angle -= add_term
        self.__insert_previous_angle()
        return self.done, self.angle

    def _blur_image(self):
        kernel_size = 3
        self.image_array = cv2.GaussianBlur(
            self.image_array, (kernel_size, kernel_size), 0)

    def _get_grayscale_image(self):
        self.image_array = cv2.cvtColor(self.image_array, cv2.COLOR_BGR2GRAY)

    def _detect_edges(self):
        self.image_array = cv2.Canny(self.image_array,
                                     50,
                                     150,
                                     )

    def _get_roi_image(self):
        xsize = self.image_array.shape[1]
        ysize = self.image_array.shape[0]
        mask = np.zeros_like(self.image_array)
        dx_bottom = int(0 * xsize)
        dx_up = int(0 * xsize)
        dy_bottom = int(0 * ysize)
        dy_up = int(0.4 * ysize)
        vertices = np.array([[(xsize - dx_bottom, ysize - dy_bottom),
                              (xsize - dx_up, dy_up),
                              (dx_up, dy_up),
                              (dx_bottom, ysize - dy_bottom)]],
                            dtype=np.int32)

        if len(self.image_array.shape) > 2:
            channel_count = self.image_array.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        self.image_array = cv2.bitwise_and(self.image_array, mask)

    def _hough_line(self):
        self.hough_lines = cv2.HoughLinesP(self.image_array,
                                           rho=1,
                                           theta=np.pi/180,
                                           threshold=25,
                                           minLineLength=20,
                                           maxLineGap=20,
                                           )
        if self.hough_lines is None:
            return

    def __draw_lines(self, lines, color):
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2 or y1 == y2:
                    continue
                cv2.line(self.original_image_array,
                         (x1, y1), (x2, y2), color, 2)

    def _get_lane_candidates(self):
        self.r_x_points, self.r_y_points = [], []
        self.l_x_points, self.l_y_points = [], []
        horizontal_clip_slope = 0.5
        for x1, y1, x2, y2 in self.hough_lines[:, 0]:
            m = self.__slope(x1, y1, x2, y2)
            if m >= horizontal_clip_slope:
                self.right.append([[x1, y1, x2, y2]])
                self.r_x_points.append(x1)
                self.r_x_points.append(x2)
                self.r_y_points.append(y1)
                self.r_y_points.append(y2)

            elif m < -horizontal_clip_slope:
                self.left.append([[x1, y1, x2, y2]])
                self.l_x_points.append(x1)
                self.l_x_points.append(x2)
                self.l_y_points.append(y1)
                self.l_y_points.append(y2)

        self.__draw_lines(self.right, RED)
        self.__draw_lines(self.left, BLUE)

    def __slope(self, x1, y1, x2, y2):
        if x1 != x2:
            return (y1 - y2) / (x1 - x2)
        else:
            return np.inf

    def _intersection_of_lanes(self):
        self.r_m, r_c = self.__one_line_linear_regression(
            self.r_x_points, self.r_y_points)
        self.l_m, l_c = self.__one_line_linear_regression(
            self.l_x_points, self.l_y_points)
        a = np.array([[-self.r_m, 1], [-self.l_m, 1]])
        b = np.array([r_c, l_c])
        try:
            self.intersection_point = np.linalg.solve(a, b)
            self.intersection_point = tuple(int(i)
                                            for i in self.intersection_point)
        except:
            print("exception")
            self.intersection_point = (self.image_array.shape[1]/2, 0)
        if self.intersection_point[0] > 0 and self.intersection_point[1] > 0:
            cv2.circle(img=self.original_image_array,
                       center=self.intersection_point,
                       radius=1,
                       color=GREEN,
                       thickness=3,
                       )

    def __one_line_linear_regression(self, x_points, y_points):
        A = np.vstack([x_points, np.ones(len(x_points))]).T
        m, c = np.linalg.lstsq(A, y_points, rcond=None)[0]
        return m, c

    def _get_angle_error(self):
        xsize = self.image_array.shape[1]
        ysize = self.image_array.shape[0]
        # < 0 when the car have to go left
        dist_to_baseline = self.intersection_point[0] - xsize/2
        dist_to_bottom = ysize - self.intersection_point[1]
        self.angle = np.arctan(dist_to_baseline / dist_to_bottom)
        cv2.line(self.original_image_array, (int(xsize/2), 0),
                 (int(xsize/2), ysize), RED, 1)


if __name__ == "__main__":
    image = cv2.imread("default_controller/sample0_7.png", cv2.IMREAD_COLOR)
    detector = LaneDetector()
    print(detector.detect_lane(image))
    # conda install -c conda-forge opencv=4.1.0
    cv2.imshow('processed', detector.image_array)
    cv2.imshow('original', detector.original_image_array)
    cv2.waitKey(0)

    image = cv2.imread("default_controller/sample0_29.png", cv2.IMREAD_COLOR)
    detector = LaneDetector()
    print(detector.detect_lane(image))
    # conda install -c conda-forge opencv=4.1.0
    cv2.imshow('processed', detector.image_array)
    cv2.imshow('original', detector.original_image_array)
    cv2.waitKey(0)
