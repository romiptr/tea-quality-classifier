import cv2
import numpy as np
import argparse
import os
import glob

def process_image(img, crop_radius, anchor_point=None):
    """Crops a circular region of interest (ROI) from an image.

    This function applies a circle mask to an image to get the ROI, 
    such as a liquid inside a cup of tea. 
    If no center point is provided, it will be automatically detected.

    args:
        img: Input image.
        radius (int): The radius of the circle ROI.
        anchor_point (x,y): A tuple (x, y) that contains a center coordinate of the circle. 
        Defaults to None, which is then automatically calculated from circle contour.

    returns:
        cropped_roi: The cropped circle ROI, squared off to the
            dimensions of the circle's bounding box.
    """

    # Set automatic anchor point if user doesn't define it
    if anchor_point == None:
        liquid_contour = find_liquid_contour(img)
        
        (x_anchor,y_anchor), radius = cv2.minEnclosingCircle(liquid_contour)
        anchor_point = (int(x_anchor),int(y_anchor))
    
    circle_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(circle_mask, anchor_point, crop_radius, 
               255, -1, lineType=cv2.LINE_AA)
    roi = cv2.bitwise_and(img, img, mask=circle_mask)

    # Crop the image to fit the ROI
    x1, y1 = (anchor_point[0] - crop_radius), (anchor_point[1] - crop_radius)
    x2, y2 = (anchor_point[0] + crop_radius), (anchor_point[1] + crop_radius)
    cropped_roi = roi[y1:y2, x1:x2]

    return cropped_roi

def find_liquid_contour(img):
    """Extract circular contour from an image mask.

    This is a helper function to get a circle contour from an image,
    the contour can then be used to help get circle ROI. 
    """

    # Get the liquid mask from hsv, where s and v is a certain thresh
    hsv =  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    cup_mask = (v_ch >= 220) & (s_ch <= 60)
    liquid_mask = (~cup_mask).astype('uint8') * 255

    # morphological operations to clean up some noise on the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    liquid_mask = cv2.morphologyEx(liquid_mask, cv2.MORPH_OPEN, 
                                   kernel, iterations=1)
    liquid_mask = cv2.morphologyEx(liquid_mask, cv2.MORPH_CLOSE, 
                                   kernel, iterations=2) 

    contours, _ = cv2.findContours(liquid_mask, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    liquid_contour = None

    # filter small blob and un-circle shaped blob from contour
    # uncircled blob is determined by the circularity threshold (0.7)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        min_area = img.shape[0]*img.shape[1]*0.01

        if (circularity >= 0.7 and area > min_area):
            liquid_contour = cnt
            break
    
    # failsave if a circle shaped blob isn't found in earlier filter, 
    # then get the largest contour area
    if liquid_contour is None and contours:
        liquid_contour = max(contours, key=cv2.contourArea)
    
    return liquid_contour

def get_parser():
    parser = argparse.ArgumentParser(description="Script for extracting circular ROI from image/s")
    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--input",
        metavar="FILE",
        help="path to a single image file",
    )
    input_group.add_argument(
        "--folder-input", 
        metavar="FOLDER_PATH",
        help="Path to folder containing images"
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=112,
        metavar="PIXEL_VALUE",
        help="Circle radius to crop image"
    )
    parser.add_argument(
        "--anchor-point",
        nargs=2,
        type=int,
        metavar=("COOR_X", "COOR_Y"),
        help="Anchor point (x,y) for center of circle crop"
    )
    parser.add_argument(
        "--output-dir",
        metavar="FOLDER_PATH",
        help="Folder path to save ROI result"
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Show output ROI in an OpenCV window"
    )
    return parser

def main():
    args = get_parser().parse_args()
    img_paths = []

    if args.folder_input:
        img_ext = ["jpg", "jpeg", "png"]
        for ext in img_ext:
            img_paths.extend(glob.glob(os.path.join(args.folder_input, ("*." + ext))))
    elif args.input:
        img_paths.append(args.input)

    print(f"Found {len(img_paths)} image/s to process from {args.folder_input or args.input}")

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image at {img_path}. Skipping.")
            continue
        
        result = process_image(img, args.radius, args.anchor_point)
        out_filename = os.path.basename(img_path).split('.')[0] + ".jpg"            
        
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, out_filename)

            cv2.imwrite(save_path, result)
            print(f"Image saved successfully to: {save_path}")

        if args.show_output:
            cv2.imshow(f"Result for {out_filename}", result)
            if cv2.waitKey(0) == ord('q'): # quit loop with 'q' key
                break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()