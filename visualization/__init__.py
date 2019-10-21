import cv2
import os
import matplotlib.image as mpimg


def draw_dots(dots_img, coordinates, color=(255, 0, 0), size=5):
    '''Draws dots at the coordinates on image
    '''
    dots_img = cv2.circle(dots_img, (coordinates[0], coordinates[1]),
                          size, color, -1)
    return dots_img


def add_text(img, text, coordinates, color=(255, 0, 0)):
    '''Adds text to the image at the coordinates specified
    '''
    img = cv2.putText(img, text, coordinates, cv2.FONT_HERSHEY_PLAIN, 1, color)
    return img


def save_result_image(img, output_path, old_filename, suffix,
                      isGrayScale=False):
    '''
    saves image to directory, attaching suffix to its name
    '''
    # check if directory exists, if not create it
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # get filename of result image
    _, tail = os.path.split(old_filename)
    (root, ext) = os.path.splitext(tail)
    result_filename = os.path.join(output_path, root + suffix + ext)

    # save the result image
    if isGrayScale:
        mpimg.imsave(result_filename, img, cmap='gray')
    else:
        mpimg.imsave(result_filename, img)
