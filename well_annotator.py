import numpy as np
import cv2
import time
import pickle
import skimage
import argparse


class WellAnnotator():

    def __init__(self, args):
        args = args
        self.img = cv2.imread(args.file)
        print(self.img.shape)
        self.grid = False
        self.annotating = False
        self.zoomed = False
        self.accept_message_displayed = False
        self.y_train = None
        self.x_train = None
        self.annotation_position_x, self.annotation_position_y = 0, 0
        self.last_x = 0
        self.last_y = 0
        self.zoomed_img = None

        self.annotated_img = np.zeros((500, 500, 1), dtype=np.uint8)

        # the values we will use to resize the image at the end to fit the screen
        self.user_view_x = int(1024 * .78)
        self.user_view_y = int(822 * .78)

        self.y_locs = list(np.arange(0, self.img.shape[0] + 822, 822))
        self.x_locs = list(np.arange(0, self.img.shape[1] + 1024, 1024))
        self.user_x_index = list(np.arange(len(self.x_locs) - 1))
        self.user_y_index = list(np.arange(len(self.y_locs) - 1))

        self.user_x_locs = list(np.arange(0, self.user_view_x, int(self.user_view_x / len(self.user_x_index))))
        self.user_y_locs = list(np.arange(0, self.user_view_y, int(self.user_view_y / len(self.user_y_index))))

        cv2.namedWindow('Annotator')
        cv2.moveWindow('Annotator', -200, 0)
        cv2.setMouseCallback('Annotator', self.mouse_event)
        self.resize_and_show_img(self.img)
        self.start_event_loop()

    def write_to_file(self):
        fn = args.file.split('.tif')[0] + '_annotated.p'
        pickle.dump((self.x_train, self.y_train), open(fn, 'wb'))

    def get_indices_from_click(self, x, y):
        x_index, y_index = 0, 0
        for loc in self.user_x_locs:
            if x > loc:
                x_index = loc
        for loc in self.user_y_locs:
            if y > loc:
                y_index = loc
        x_index = self.user_x_locs.index(x_index)
        y_index = self.user_y_locs.index(y_index)
        # print(x_index,y_index)
        return x_index, y_index

    def get_zoomed_slice(self, x_index, y_index):
        start_y = self.x_locs[x_index]
        start_x = self.y_locs[y_index]
        end_y = self.x_locs[x_index + 1]
        end_x = self.y_locs[y_index + 1]
        self.zoomed_img = np.copy(self.img[start_x:end_x,
                                  start_y:end_y, :])
        return self.zoomed_img

    def resize_mouse_position(self, x, y):
        annotation_position_x = x / self.user_view_x * self.annotated_img.shape[0]
        annotation_position_y = y / self.user_view_y * self.annotated_img.shape[1]
        return int(annotation_position_x), int(annotation_position_y)

    def annotate(self, event, x, y, flags, param):
        # event = 1 on click, 0 on moving
        # flag = 1 as long as mouse is held down
        fill_val = 255
        self.annotation_position_x, self.annotation_position_y = self.resize_mouse_position(x, y)
        # last_x,last_y = -1,-1
        if event == 1 and flags == 1:
            self.annotated_img[self.annotation_position_y, self.annotation_position_x] = fill_val
            self.last_x = self.annotation_position_x
            self.last_y = self.annotation_position_y
            cv2.imshow('Annotation', self.annotated_img)
        elif event == 0 and flags == 1:
            self.annotated_img[self.annotation_position_y, self.annotation_position_x] = fill_val
            lineThickness = 2
            cv2.line(self.annotated_img, (self.last_x, self.last_y),
                     (self.annotation_position_x, self.annotation_position_y), fill_val, lineThickness)
            # cv2.line(mask_display, (last_x, last_y), (x, y), (fill_val, fill_val, fill_val), lineThickness)
            self.last_x = self.annotation_position_x
            self.last_y = self.annotation_position_y
            cv2.imshow('Annotation', self.annotated_img)
        elif event == 4:
            self.annotated_img = self.fill_contour(self.annotated_img)
            cv2.imshow('Annotation', self.annotated_img)

    def mouse_event(self, event, x, y, flags, param):
        # we need to find where the user clicked within the grid
        if event == 1 and not self.annotating and not self.zoomed:
            self.zoomed = True
            x_index, y_index = self.get_indices_from_click(x, y)
            zoomed_img = self.get_zoomed_slice(x_index, y_index)
            self.resize_and_show_img(zoomed_img)
        if self.annotating and self.zoomed:
            self.annotate(event, x, y, flags, param)

    def fill_contour(self, mask):
        fill_val = 255
        im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.drawContours(mask, [cnt], 0, fill_val, -1)
        return mask

    def resize_and_show_img(self, img):
        img_resized = cv2.resize(img, (self.user_view_x, self.user_view_y), cv2.INTER_AREA)
        cv2.imshow('Annotator', img_resized)

    def do_nothing(self):
        time.sleep(.01)
        if cv2.getWindowProperty('Annotator', 0) < 0:
            exit()

    def end_program(self):
        cv2.destroyAllWindows()
        self.write_to_file()
        exit()

    def toggle_grid(self):
        if self.annotating: return
        self.zoomed = False
        self.grid = not self.grid
        if self.grid:
            self.turn_on_grid()
        else:
            self.turn_off_grid()

    def turn_on_grid(self):
        grid_img = self.create_grid_img(self.img)
        self.resize_and_show_img(grid_img)

    def turn_off_grid(self):
        img = self.img
        self.resize_and_show_img(img)

    def turn_on_annotation(self):
        cv2.namedWindow('Annotation')
        cv2.moveWindow("Annotation", 1000, 0)
        cv2.imshow('Annotation', self.annotated_img)

    def toggle_annotation(self):
        if self.zoomed: self.annotating = not self.annotating
        if self.annotating:
            self.turn_on_annotation()
        elif not self.annotating:
            self.display_accept_message()

    def display_accept_message(self):
        self.accept_message_displayed = True
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (50, 500)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2
        accept_message = np.copy(self.annotated_img)
        cv2.putText(accept_message, 'Accept? [y/n]',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        cv2.imshow('Annotation', accept_message)

    def add_to_y_train(self):
        if self.y_train is None:
            self.y_train = np.expand_dims(self.annotated_img, axis = 0)
        else:
            self.y_train = np.concatenate((self.y_train, np.expand_dims(self.annotated_img, axis = 0)), axis= 0)
        self.annotated_img = np.zeros((500, 500, 1), dtype=np.uint8)

    def add_to_x_train(self):
        img = cv2.cvtColor(self.zoomed_img, cv2.COLOR_BGR2GRAY)
        img = skimage.transform.resize(img, (500, 500), anti_aliasing=True)
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        if self.x_train is None:
            self.x_train = img
        else:
            self.x_train = np.concatenate((self.x_train, img), axis=0)

    def back_to_finding_mode(self):
        self.annotating = False
        cv2.destroyWindow('Annotation')

    def reset_annotation(self):
        self.annotating = True
        self.annotated_img = np.zeros((500, 500, 1), dtype=np.uint8)
        cv2.imshow('Annotation', self.annotated_img)
        self.accept_message_displayed = False

    def show_results(self):
        cv2.imshow('x', self.x_train[-1,:, :,0])
        cv2.imshow('y', self.y_train[-1,:, :,0])

    def accept_annotation(self):
        if self.accept_message_displayed:
            self.add_to_x_train()
            self.add_to_y_train()
            self.back_to_finding_mode()
            self.show_results()
        self.accept_message_displayed = False

    def reject_annotation(self):
        if self.accept_message_displayed:
            self.reset_annotation()

    def start_event_loop(self):
        event_dict = {
            -1: self.do_nothing,
            ord('e'): self.end_program,
            ord('g'): self.toggle_grid,
            ord('a'): self.toggle_annotation,
            ord('y'): self.accept_annotation,
            ord('n'): self.reject_annotation,
            ord('w'): self.write_to_file
        }
        while True:
            key = cv2.waitKey(5)
            if key in event_dict.keys():
                event_dict[key]()

    def create_grid_img(self, img):
        grid_img = np.copy(img)
        for x in self.x_locs:
            cv2.line(grid_img, (x, self.y_locs[0]), (x, self.y_locs[-1]), (255, 255, 255), 25)
        for y in self.y_locs:
            cv2.line(grid_img, (self.x_locs[0], y), (self.x_locs[-1], y), (255, 255, 255), 25)

        return grid_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Annotate a well for model training. \n
    \n
    Controls:\n
    g: show grid\n
    mouse click in grid to view 100x image of that grid location\n
    a: popup annotation window\n
    click and drag to circle cells of interest, closed contours will be filled\n
    a again to bring up accept or deny annotation menu\n
    y if annotation is good,\n
    n to reset annotation\n
    g again to reset view to whole-well\n
    g again to show grid \n
    repeat as necessary until all cells have been found and annotated\n
    w: WRITE BEFORE EXITING!
    ''')
    parser.add_argument('-f', '--file', dest='file', help='The well image to annotate.)')
    args = parser.parse_args()
    wa = WellAnnotator(args)
