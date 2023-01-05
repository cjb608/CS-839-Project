import gym
from gym import spaces
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt


class DrawEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.max_steps = 2
        self.current_step = 0
        self.num_classes = 8
        # target class: 0 -> '-', 1 -> '|', 2 -> '\', 3 -> '/',
        #               4 -> '^', 5 -> '<', 6 -> '>', 7 -> '+'
        self.target_class = np.random.randint(0, self.num_classes)
        # actions = [x0, y0, x1, y1]
        self.action_space = spaces.MultiDiscrete([64, 64, 64, 64])
        # state = [target class, current - 1 x0, current - 1 y0, current - 1 x1, current - 1 y1,
        #                        current - 0 x0, current - 0 y0, current - 0 x1, current - 0 y1]
        self.observation_space = spaces.MultiDiscrete([8, 64, 64, 64, 64,
                                                          64, 64, 64, 64])
        self.state = np.array([self.target_class, 0, 0, 0, 0,
                                                  0, 0, 0, 0])
        self.reward = 0
        self.done = False
        self.line_segments = np.zeros((2, 2, 2))

    def reset(self):
        self.done = False
        self.current_step = 0
        self.reward = 0
        self.target_class = np.random.randint(0, self.num_classes)
        self.state = np.array([self.target_class, 0, 0, 0, 0,
                                                  0, 0, 0, 0])
        return self.state

    def step(self, action):
        self.state[1:5] = self.state[5:9]
        self.state[5:9] = action
        self.current_step += 1

        self.line_segments = np.reshape(self.state[1:], (2, 2, 2))

        if self.current_step == 1:
            if self.target_class == 0:
                if self.is_horizontal_line(self.line_segments[1, :, :]):
                    self.reward = 1
                    self.done = True
                else:
                    self.reward = -1
                    self.done = True

            if self.target_class == 1:
                if self.is_vertical_line(self.line_segments[1, :, :]):
                    self.reward = 1
                    self.done = True
                else:
                    self.reward = -1
                    self.done = True

            if self.target_class == 2:
                if self.is_back_slash(self.line_segments[1, :, :]):
                    self.reward = 1
                    self.done = True
                else:
                    self.reward = -1
                    self.done = True

            if self.target_class == 3:
                if self.is_forward_slash(self.line_segments[1, :, :]):
                    self.reward = 1
                    self.done = True
                else:
                    self.reward = -1
                    self.done = True

            if self.target_class == 4:
                if self.is_forward_slash(self.line_segments[1, :, :]):
                    self.reward = 0.5
                elif self.is_back_slash(self.line_segments[1, :, :]):
                    self.reward = 0.5
                else:
                    self.reward = 0

            if self.target_class == 5:
                if self.is_forward_slash(self.line_segments[1, :, :]):
                    self.reward = 0.5
                elif self.is_back_slash(self.line_segments[1, :, :]):
                    self.reward = 0.5
                else:
                    self.reward = 0

            if self.target_class == 6:
                if self.is_forward_slash(self.line_segments[1, :, :]):
                    self.reward = 0.5
                elif self.is_back_slash(self.line_segments[1, :, :]):
                    self.reward = 0.5
                else:
                    self.reward = 0

            if self.target_class == 7:
                if self.is_horizontal_line(self.line_segments[1, :, :]):
                    self.reward = 0.5
                elif self.is_vertical_line(self.line_segments[1, :, :]):
                    self.reward = 0.5
                else:
                    self.reward = 0

        if self.current_step == 2:

            if self.target_class == 4:
                if self.is_carrot(self.line_segments[0, :, :], self.line_segments[1, :, :]):
                    self.reward = 0.5
                    self.done = True
                else:
                    self.reward = -1
                    self.done = True

            if self.target_class == 5:
                if self.is_less_than(self.line_segments[0, :, :], self.line_segments[1, :, :]):
                    self.reward = 0.5
                    self.done = True
                else:
                    self.reward = -1
                    self.done = True

            if self.target_class == 6:
                if self.is_greater_than(self.line_segments[0, :, :], self.line_segments[1, :, :]):
                    self.reward = 0.5
                    self.done = True
                else:
                    self.reward = -1
                    self.done = True

            if self.target_class == 7:
                if self.is_plus(self.line_segments[0, :, :], self.line_segments[1, :, :]):
                    self.reward = 0.5
                    self.done = True
                else:
                    self.reward = -1
                    self.done = True

        if self.current_step >= self.max_steps:
            self.done = True

        return self.state, self.reward, self.done, {}

    def render(self):
        img = np.zeros((64, 64))
        shape_list = np.array(['-', '|', chr(92), '/', '^', '<', '>', '+', 'L', 'T', 'V', 'X'])
        target_shape = shape_list[self.state[0]]
        if self.current_step == 1:
            img = cv2.line(img, (self.state[5], self.state[6]), (self.state[7], self.state[8]), 255, 1)
        elif self.current_step == 2:
            img = cv2.line(img, (self.state[1], self.state[2]), (self.state[3], self.state[4]), 255, 1)
            img = cv2.line(img, (self.state[5], self.state[6]), (self.state[7], self.state[8]), 255, 1)
        plt.imshow(img, origin='lower', cmap=plt.cm.gray_r)
        plt.title('Target = {t} \n Reward = {r}'.format(t=target_shape, r=self.reward))
        plt.show()

    def is_horizontal_line(self, line_segment):
        """
        function that returns true if the line segment has a
        length greater than 10 and an angle of 0 or 180 degrees
        """
        dist = self.distance_between_two_points(line_segment)
        angle = self.angle_of_line_segment(line_segment)
        if dist >= 10:
            if 0 <= angle <= 10 or 170 <= angle <= 190:
                return True
        return False

    def is_vertical_line(self, line_segment):
        """
        function that returns True if the line segment has a
        length greater than 10 and an angle of 90 degrees
        """
        dist = self.distance_between_two_points(line_segment)
        angle = self.angle_of_line_segment(line_segment)
        if dist >= 10:
            if 80 <= angle <= 100 or 260 <= angle <= 280:
                return True
        return False

    def is_back_slash(self, line_segment):
        """
        function that returns True if the line segment has a
        length greater than 5 and an angle of 135 degrees
        """
        dist = self.distance_between_two_points(line_segment)
        angle = self.angle_of_line_segment(line_segment)
        if dist >= 10:
            if 110 <= angle <= 160 or 290 <= angle <= 340:
                return True
        return False

    def is_forward_slash(self, line_segment):
        """
        function that returns True if the line segment has a
        length greater than 10 and an angle of 45 degrees
        """
        dist = self.distance_between_two_points(line_segment)
        angle = self.angle_of_line_segment(line_segment)
        if dist >= 10:
            if 20 <= angle <= 70 or 200 <= angle <= 250:
                return True
        return False

    def is_carrot(self, line_segment1, line_segment2):
        """
        function that returns True if 2 line segments intersect,
        share a common point that has the largest y value,
        1 line segment is a back slash and the other line segment is a forward slash,
        and the length of each line is equal
        """
        intersect = self.intersect(line_segment1, line_segment2)
        equal_point, pos = self.equal_points(line_segment1, line_segment2)
        back_slash1 = self.is_back_slash(line_segment1)
        back_slash2 = self.is_back_slash(line_segment2)
        forward_slash1 = self.is_forward_slash(line_segment1)
        forward_slash2 = self.is_forward_slash(line_segment2)
        dist1 = self.distance_between_two_points(line_segment1)
        dist2 = self.distance_between_two_points(line_segment2)
        if intersect:
            if equal_point:
                if pos == 2:
                    if (back_slash1 and forward_slash2) or (back_slash2 and forward_slash1):
                        if dist2 - 10 <= dist1 <= dist2 + 10 or dist1 - 10 <= dist2 <= dist1 + 10:
                            return True
        return False

    def is_less_than(self, line_segment1, line_segment2):
        """
        function that returns True if 2 line segments intersect,
        share a common point that has the smallest x value,
        1 line segment is a back slash and the other line segment is a forward slash,
        and the length of each line is equal
        """
        intersect = self.intersect(line_segment1, line_segment2)
        equal_point, pos = self.equal_points(line_segment1, line_segment2)
        back_slash1 = self.is_back_slash(line_segment1)
        back_slash2 = self.is_back_slash(line_segment2)
        forward_slash1 = self.is_forward_slash(line_segment1)
        forward_slash2 = self.is_forward_slash(line_segment2)
        dist1 = self.distance_between_two_points(line_segment1)
        dist2 = self.distance_between_two_points(line_segment2)
        if intersect:
            if equal_point:
                if pos == 0:
                    if (back_slash1 and forward_slash2) or (back_slash2 and forward_slash1):
                        if dist2 - 10 <= dist1 <= dist2 + 10 or dist1 - 10 <= dist2 <= dist1 + 10:
                            return True
        return False

    def is_greater_than(self, line_segment1, line_segment2):
        """
        function that returns True if 2 line segments intersect,
        share a common point that has the largest x value,
        1 line segment is a back slash and the other line segment is a forward slash,
        and the length of each line is equal
        """
        intersect = self.intersect(line_segment1, line_segment2)
        equal_point, pos = self.equal_points(line_segment1, line_segment2)
        back_slash1 = self.is_back_slash(line_segment1)
        back_slash2 = self.is_back_slash(line_segment2)
        forward_slash1 = self.is_forward_slash(line_segment1)
        forward_slash2 = self.is_forward_slash(line_segment2)
        dist1 = self.distance_between_two_points(line_segment1)
        dist2 = self.distance_between_two_points(line_segment2)
        if intersect:
            if equal_point:
                if pos == 1:
                    if (back_slash1 and forward_slash2) or (back_slash2 and forward_slash1):
                        if dist2 - 10 <= dist1 <= dist2 + 10 or dist1 - 10 <= dist2 <= dist1 + 10:
                            return True
        return False

    def is_plus(self, line_segment1, line_segment2):
        """
        function that returns True if 2 line segments intersect,
        do not share a common point,
        1 line segment is a horizontal line and the other line segment is a vertical line,
        the midpoints of the 2 line segments are equal,
        and the length of each line is equal
        """
        intersect = self.intersect(line_segment1, line_segment2)
        equal_point, pos = self.equal_points(line_segment1, line_segment2)
        horizontal1 = self.is_horizontal_line(line_segment1)
        horizontal2 = self.is_horizontal_line(line_segment2)
        vertical1 = self.is_vertical_line(line_segment1)
        vertical2 = self.is_vertical_line(line_segment2)
        midpoint1 = self.midpoint_of_line_segment(line_segment1)
        midpoint2 = self.midpoint_of_line_segment(line_segment2)
        midpoint_dist = self.distance_between_two_points(np.array([midpoint1, midpoint2]))
        dist1 = self.distance_between_two_points(line_segment1)
        dist2 = self.distance_between_two_points(line_segment2)
        if intersect:
            if not equal_point:
                if (horizontal1 and vertical2) or (horizontal2 and vertical1):
                    if midpoint_dist < 6:
                        if dist2 - 10 <= dist1 <= dist2 + 10 or dist1 - 10 <= dist2 <= dist1 + 10:
                            return True
        return False

    def distance_between_two_points(self, line_segment):
        """
        function that returns the distance
        between two points of a line segment
        """
        dist = ((line_segment[1, 0] - line_segment[0, 0])**2 + (line_segment[1, 1] - line_segment[0, 1])**2)**0.5
        return dist

    def angle_of_line_segment(self, line_segment):
        """
        function that returns the angle
        between a line segment and the x-axis
        """
        m = (line_segment[1, 1] - line_segment[0, 1]) / (line_segment[1, 0] - line_segment[0, 0] + 0.000001)
        angle = math.atan(m)
        angle = math.degrees(angle)
        if angle < 0:
            angle = 180 + angle
        return angle

    def midpoint_of_line_segment(self, line_segment):
        """
        function that returns the midpoint of a line segment
        """
        midpoint = np.array(((line_segment[0, 0] + line_segment[1, 0]) / 2,
                            (line_segment[0, 1] + line_segment[1, 1]) / 2))
        return midpoint

    def on_segment(self, point1, point2, point3):
        """
        function that returns True if point2
        lies on the line segment point1 -> point2
        """
        if (max(point1[0], point3[0]) >= point2[0] >= min(point1[0], point3[0]) and
                max(point1[1], point3[1]) >= point2[1] >= min(point1[1], point3[1])):
            return True
        else:
            return False

    def orientation(self, point1, point2, point3):
        """
        function that returns 0 if the 3 points are collinear,
        1 if the 3 points are clockwise,
        or 2 if the three points are counterclockwise

        """
        val = (float(point2[1] - point1[1]) * (point3[0] - point2[0]) -
               float(point2[0] - point1[0]) * (point3[1] - point2[1]))

        if val > 0:
            return 1
        elif val < 0:
            return 2
        else:
            return 0

    def intersect(self, line_segment1, line_segment2):
        """
        function that returns True if line_segment1
        and line_segment2 intersect
        """
        o1 = self.orientation(line_segment1[0, :], line_segment1[1, :], line_segment2[0, :])
        o2 = self.orientation(line_segment1[0, :], line_segment1[1, :], line_segment2[1, :])
        o3 = self.orientation(line_segment2[0, :], line_segment2[1, :], line_segment1[0, :])
        o4 = self.orientation(line_segment2[0, :], line_segment2[1, :], line_segment1[1, :])

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and self.on_segment(line_segment1[0, :], line_segment2[0, :], line_segment1[1, :]):
            return True

        if o2 == 0 and self.on_segment(line_segment1[0, :], line_segment2[1, :], line_segment1[1, :]):
            return True

        if o3 == 0 and self.on_segment(line_segment2[0, :], line_segment1[0, :], line_segment2[1, :]):
            return True

        if o4 == 0 and self.on_segment(line_segment2[1, :], line_segment1[1, :], line_segment2[1, :]):
            return True

        return False

    def equal_points(self, line_segment1, line_segment2):
        """
        function that returns True if 2 line segments share a common point and
        returns where the two equal points are in relationship to the two other points
        (0 -> left, 1 -> right, 2 -> top, 3 -> bottom,
        4 -> top left, 5 -> bottom left, 6 -> top right, 7 -> bottom right,
        8 -> none)
        """
        if np.array_equal(line_segment1[0, :], line_segment2[0, :]):
            if line_segment1[0, 0] < np.min((line_segment1[1, 0], line_segment2[1, 0])):
                return True, 0
            if line_segment1[0, 0] > np.max((line_segment1[1, 0], line_segment2[1, 0])):
                return True, 1
            if line_segment1[0, 1] > np.max((line_segment1[1, 1], line_segment2[1, 1])):
                return True, 2
            if line_segment1[0, 1] < np.min((line_segment1[1, 1], line_segment2[1, 1])):
                return True, 3
            if (line_segment1[0, 0] == np.min((line_segment1[1, 0], line_segment2[1, 0])) and
                    line_segment1[0, 1] == np.max((line_segment1[1, 1], line_segment2[1, 1]))):
                return True, 4
            if (line_segment1[0, 0] == np.min((line_segment1[1, 0], line_segment2[1, 0])) and
                    line_segment1[0, 1] == np.min((line_segment1[1, 1], line_segment2[1, 1]))):
                return True, 5
            if (line_segment1[0, 0] == np.max((line_segment1[1, 0], line_segment2[1, 0])) and
                    line_segment1[0, 1] == np.max((line_segment1[1, 1], line_segment2[1, 1]))):
                return True, 6
            if (line_segment1[0, 0] == np.max((line_segment1[1, 0], line_segment2[1, 0])) and
                    line_segment1[0, 1] == np.min((line_segment1[1, 1], line_segment2[1, 1]))):
                return True, 6
            return True, 8

        if np.array_equal(line_segment1[0, :], line_segment2[1, :]):
            if line_segment1[0, 0] < np.min((line_segment1[1, 0], line_segment2[0, 0])):
                return True, 0
            if line_segment1[0, 0] > np.max((line_segment1[1, 0], line_segment2[0, 0])):
                return True, 1
            if line_segment1[0, 1] > np.max((line_segment1[1, 1], line_segment2[0, 1])):
                return True, 2
            if line_segment1[0, 1] < np.min((line_segment1[1, 1], line_segment2[0, 1])):
                return True, 3
            if (line_segment1[0, 0] == np.min((line_segment1[1, 0], line_segment2[0, 0])) and
                    line_segment1[0, 1] == np.max((line_segment1[1, 1], line_segment2[0, 1]))):
                return True, 4
            if (line_segment1[0, 0] == np.min((line_segment1[1, 0], line_segment2[0, 0])) and
                    line_segment1[0, 1] == np.min((line_segment1[1, 1], line_segment2[0, 1]))):
                return True, 5
            if (line_segment1[0, 0] == np.max((line_segment1[1, 0], line_segment2[0, 0])) and
                    line_segment1[0, 1] == np.max((line_segment1[1, 1], line_segment2[0, 1]))):
                return True, 6
            if (line_segment1[0, 0] == np.max((line_segment1[1, 0], line_segment2[0, 0])) and
                    line_segment1[0, 1] == np.min((line_segment1[1, 1], line_segment2[0, 1]))):
                return True, 6
            return True, 8

        if np.array_equal(line_segment1[1, :], line_segment2[0, :]):
            if line_segment1[1, 0] < np.min((line_segment1[0, 0], line_segment2[1, 0])):
                return True, 0
            if line_segment1[1, 0] > np.max((line_segment1[0, 0], line_segment2[1, 0])):
                return True, 1
            if line_segment1[1, 1] > np.max((line_segment1[0, 1], line_segment2[1, 1])):
                return True, 2
            if line_segment1[1, 1] < np.min((line_segment1[0, 1], line_segment2[1, 1])):
                return True, 3
            if (line_segment1[1, 0] == np.min((line_segment1[0, 0], line_segment2[1, 0])) and
                    line_segment1[1, 1] == np.max((line_segment1[0, 1], line_segment2[1, 1]))):
                return True, 4
            if (line_segment1[1, 0] == np.min((line_segment1[0, 0], line_segment2[1, 0])) and
                    line_segment1[1, 1] == np.min((line_segment1[0, 1], line_segment2[1, 1]))):
                return True, 5
            if (line_segment1[1, 0] == np.max((line_segment1[0, 0], line_segment2[1, 0])) and
                    line_segment1[1, 1] == np.max((line_segment1[0, 1], line_segment2[1, 1]))):
                return True, 6
            if (line_segment1[1, 0] == np.max((line_segment1[0, 0], line_segment2[1, 0])) and
                    line_segment1[1, 1] == np.min((line_segment1[0, 1], line_segment2[1, 1]))):
                return True, 6
            return True, 8

        if np.array_equal(line_segment1[1, :], line_segment2[1, :]):
            if line_segment1[1, 0] < np.min((line_segment1[0, 0], line_segment2[0, 0])):
                return True, 0
            if line_segment1[1, 0] > np.max((line_segment1[0, 0], line_segment2[0, 0])):
                return True, 1
            if line_segment1[1, 1] > np.max((line_segment1[0, 1], line_segment2[0, 1])):
                return True, 2
            if line_segment1[1, 1] < np.min((line_segment1[0, 1], line_segment2[0, 1])):
                return True, 3
            if (line_segment1[1, 0] == np.min((line_segment1[0, 0], line_segment2[0, 0])) and
                    line_segment1[1, 1] == np.max((line_segment1[0, 1], line_segment2[0, 1]))):
                return True, 4
            if (line_segment1[1, 0] == np.min((line_segment1[0, 0], line_segment2[0, 0])) and
                    line_segment1[1, 1] == np.min((line_segment1[0, 1], line_segment2[0, 1]))):
                return True, 5
            if (line_segment1[1, 0] == np.max((line_segment1[0, 0], line_segment2[0, 0])) and
                    line_segment1[1, 1] == np.max((line_segment1[0, 1], line_segment2[0, 1]))):
                return True, 6
            if (line_segment1[1, 0] == np.max((line_segment1[0, 0], line_segment2[0, 0])) and
                    line_segment1[1, 1] == np.min((line_segment1[0, 1], line_segment2[0, 1]))):
                return True, 6
            return True, 8

        return False, 8
