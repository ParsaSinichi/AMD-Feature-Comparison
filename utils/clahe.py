import os
import cv2
import argparse
def apply_clahe(img, clip_limit=2.0, grid_size=(8, 8)):

    clahe = cv2.createCLAHE(clip_limit, grid_size)

    b, g, r = cv2.split(img)
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)

    img_clahe = cv2.merge((b, g, r))
    return img_clahe

def process_images(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for folder_type in ["Train", "Test"]:
        input_folder_path = os.path.join(input_folder, folder_type)
        output_folder_path = os.path.join(output_folder, folder_type)

        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        for subfolder in os.listdir(input_folder_path):
            subfolder_path = os.path.join(input_folder_path, subfolder)
            output_subfolder_path = os.path.join(output_folder_path, subfolder)

            if not os.path.isdir(subfolder_path):
                continue

            if not os.path.exists(output_subfolder_path):
                os.makedirs(output_subfolder_path)

            for filename in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, filename)
                output_image_path = os.path.join(
                    output_subfolder_path, filename)

                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Error: Unable to read image '{image_path}'")
                        continue

                    # Apply CLAHE
                    processed_image = apply_clahe(image)

                    cv2.imwrite(output_image_path, processed_image)
                except Exception as e:
                    print(f"Error processing image '{image_path}': {str(e)}")

    print("Processing completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply CLAHE to images in a folder structure")
    parser.add_argument("--input_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    args = parser.parse_args()

    process_images(args.input_folder, args.output_folder)
