import random

from .identity import Identity
from .crop import Crop, Cropout, Dropout, CropResize, CropIn
from .gaussian_noise import GN
from .middle_filter import MF
from .gaussian_filter import GF
from .salt_pepper_noise import SP
from .resize import Resize, ResizeCropout
from .jpeg import Jpeg, JpegSS, JpegMask, JpegTest, JpegTwice
from .combined import Combined
from .roll import Roll, RollTest
