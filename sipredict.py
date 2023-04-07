from PIL import Image

from cea import CEA

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


if __name__ == "__main__":
    ceanet = CEA()
    #----------------------------------------------------------------------------------------------------------#
    #   'predict' or 'dir_predict'
    #----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    crop            = False
    #-------------------------------------------------------------------------#
    #   dir_origin_path: read eddy test image
    #   dir_save_path: save predict result
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"

    if mode == "predict":
      
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = ceanet.detect_image(image, crop = crop)
                r_image.show()

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = ceanet.detect_image(image, corflag = 1)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")