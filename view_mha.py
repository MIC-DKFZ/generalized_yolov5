from medpy.io import load
import matplotlib.pyplot as plt

# path = "D:/Datasets/DKFZ/Node21/nodule_patches/1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860_dcm_0.mha"
path = r"D:\Datasets\DKFZ\Node21\proccessed_data\images\c0001.mha"
im, im_header = load(path)
print(im.shape)
plt.imshow(im)
plt.show()