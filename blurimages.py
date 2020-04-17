from PIL import Image, ImageFilter
import numpy as np

class BlurImage:


    def blurImage(self, input_image_path, scale, blur_radius):
        image1 = Image.open(input_image_path)
        w, h = image1.size
        image2 = image1.resize((int(w/scale), int(h/scale)), Image.ANTIALIAS)
        image3 = image2.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return image3

    #incase we want to blur all images in a folder
    def blurAll(self, path, ext, batch_start, num_images):
        images = []
        for i in range(batch_start, batch_start + num_images):
            temp = blurImage(path + "/" + str(i) + "." + ext)
            images.append(temp)
        return np.array(temp)


a = BlurImage()
a.blurImage("utsav-shah-R1Z2tCpl9Zk-unsplash.jpg", 16, 0.5).show()
