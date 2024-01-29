import os
import numpy as np
from src.keras_utils import load_model
import cv2
from src.keras_utils import detect_lp_width
from src.utils import im2single
from src.drawing_utils import draw_losangle
import argparse
import time

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        required=True,
        help="Input folder containing images",
    )
    parser.add_argument(
        "-v",
        "--vtype",
        type=str,
        default="fullimage",
        help="Image type (car, truck, bus, bike, or fullimage)",
    )
    parser.add_argument(
        "-t", "--lp_threshold", type=float, default=0.35, help="Detection Threshold"
    )
    args = parser.parse_args()

    # Parameters of the method
    lp_threshold = args.lp_threshold
    ocr_input_size = [80, 240]  # desired LP size (width x height)

    # Loads network and weights
    iwpod_net = load_model("weights/iwpod_net")

    # Iterate through images in the input folder
    image_files = os.listdir(args.input_folder)
    for image_file in image_files:
        print(image_file)
        image_path = os.path.join(args.input_folder, image_file)

        # Loads image with vehicle crop or full image with vehicle(s) roughly framed.
        Ivehicle = cv2.imread(image_path)
        vtype = args.vtype
        iwh = np.array(Ivehicle.shape[1::-1], dtype=float).reshape((2, 1))

        if vtype in ["car", "bus", "truck"]:
            # Defines crops for car, bus, truck based on input aspect ratio
            ASPECTRATIO = max(1, min(2.75, 1.0 * Ivehicle.shape[1] / Ivehicle.shape[0]))
            WPODResolution = 256
            lp_output_resolution = tuple(ocr_input_size[::-1])
        elif vtype == "fullimage":
            # Defines crop if vehicles were not cropped
            ASPECTRATIO = 1
            WPODResolution = 480
            lp_output_resolution = tuple(ocr_input_size[::-1])
        else:
            # Defines crop for motorbike
            ASPECTRATIO = 1.0
            WPODResolution = 208
            lp_output_resolution = (int(1.5 * ocr_input_size[0]), ocr_input_size[0])

        # Runs IWPOD-NET. Returns list of LP data and cropped LP images
        Llp, LlpImgs, _ = detect_lp_width(
            iwpod_net,
            im2single(Ivehicle),
            WPODResolution * ASPECTRATIO,
            2**4,
            lp_output_resolution,
            lp_threshold,
        )
        for i, img in enumerate(LlpImgs):
            print("Rectifying...")
            # Draws LP quadrilateral in input image
            pts = Llp[i].pts * iwh
            draw_losangle(Ivehicle, pts, color=(0, 0, 255.0), thickness=2)

            # Shows each detected LP
            # cv2.imshow('Rectified plate %d' % i, img)
            img = cv2.convertScaleAbs(img, alpha=(255.0))
            cv2.imwrite(f"results_tnip/rectified_iwpod_{image_file}", img)

        # Shows original image with detected plates (quadrilateral)
        # cv2.imshow('Image and LPs', Ivehicle)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
    end = time.time()
    total_time = end - start
    print(f"Total time: {total_time}")
