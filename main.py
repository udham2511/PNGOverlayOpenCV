import numpy
import cv2


def overlay(
    background: numpy.ndarray, foreground: numpy.ndarray, point: tuple[int]
) -> None:
    """overlay PNG image to another png image

    Args:
        background (numpy.ndarray): background image
        foreground (numpy.ndarray): foreground image
        point (tuple[int]): position of foreground image
    """
    FGX = max(0, point[0] * -1)
    BGX = max(0, point[0])
    BGY = max(0, point[1])
    FGY = max(0, point[1] * -1)

    BGH, BGW = background.shape[:2]
    FGH, FGW = foreground.shape[:2]

    W = min(FGW, BGW, FGW + point[0], BGW - point[0])
    H = min(FGH, BGH, FGH + point[1], BGH - point[1])

    foreground = foreground[FGY : FGY + H, FGX : FGX + W]
    backgroundSubSection = background[BGY : BGY + H, BGX : BGX + W]

    alphaMask = numpy.dstack(tuple(foreground[:, :, 3] / 255.0 for _ in range(3)))

    background[BGY : BGY + H, BGX : BGX + W] = (
        backgroundSubSection * (1 - alphaMask) + foreground[:, :, :3] * alphaMask
    )


SHAPE = (600, 600, 3)

screen = numpy.zeros(SHAPE, numpy.uint8)
screen += 255

RCIRCLE = cv2.imread(r"./resources/rCircle.png", cv2.IMREAD_UNCHANGED)
GCIRCLE = cv2.imread(r"./resources/gCircle.png", cv2.IMREAD_UNCHANGED)
BCIRCLE = cv2.imread(r"./resources/bCircle.png", cv2.IMREAD_UNCHANGED)

overlay(screen, RCIRCLE, (72, 100))
overlay(screen, GCIRCLE, (272, 100))
overlay(screen, BCIRCLE, (172, 270))

cv2.imshow("Overlay PNG", screen)

cv2.waitKey(0)
cv2.destroyAllWindows()
