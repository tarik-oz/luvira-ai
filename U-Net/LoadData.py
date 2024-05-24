import cv2
import numpy as np
import glob
from tqdm import tqdm

Height = 256
Width = 256


path = "C:/Users/Tarik/Desktop/U-Net_conda/data/"
imagesPath = path + "images/*.jpg"
masksPath = path + "masks/*.jpg"

listOfImages = glob.glob(imagesPath)
listOfMasks = glob.glob(masksPath)

print(len(listOfImages))
print(len(listOfMasks))

img = cv2.imread(listOfImages[0], cv2.IMREAD_COLOR)
print(img.shape)
img = cv2.resize(img, (Width, Height))
print(img.shape)

mask = cv2.imread(listOfMasks[0], cv2.IMREAD_GRAYSCALE)
print(mask.shape)
mask = cv2.resize(mask, (Width, Height))
print(mask.shape)

# cv2.imshow("img", img)
# cv2.imshow("mask", mask)

cv2.waitKey(0)

mask16 = cv2.resize(mask, (16, 16))
print(mask16)


mask16[mask16 > 0] = 1

print("=============")
print(mask16)

allImages = []
maskImages = []

print("start loading the train images and masks")

for imgFile, maskFile in tqdm(zip(listOfImages, listOfMasks), total=len(listOfImages)):
    img = cv2.imread(imgFile, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (Width, Height))
    img = img / 255.0
    img = img.astype(np.float32)
    allImages.append(img)

    mask = cv2.imread(maskFile, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (Width, Height))

    mask[mask > 0] = 1
    maskImages.append(mask)

allImagesNP = np.array(allImages)
maskImagesNP = np.array(maskImages)
maskImagesNP = maskImagesNP.astype(int)

print("Shapes of train images and masks :")
print(allImagesNP.shape)
print(maskImagesNP.shape)

from sklearn.model_selection import train_test_split

split = 0.1

train_imgs, valid_imgs = train_test_split(allImagesNP, test_size=split, random_state=42)
train_masks, valid_masks = train_test_split(
    maskImagesNP, test_size=split, random_state=42
)

print("Shapes of train images and masks: ")
print(train_imgs.shape)
print(train_masks.shape)

print("Shapes of validation images and masks: ")
print(valid_imgs.shape)
print(valid_masks.shape)

print("Save the data: ")
np.save(
    "C:/Users/Tarik/Desktop/U-Net_conda/data/Unet_Train_Hair_Images.npy", train_imgs
)
np.save(
    "C:/Users/Tarik/Desktop/U-Net_conda/data/Unet_Train_Hair_Masks.npy", train_masks
)
np.save(
    "C:/Users/Tarik/Desktop/U-Net_conda/data/Unet_Valid_Hair_Images.npy", valid_imgs
)
np.save(
    "C:/Users/Tarik/Desktop/U-Net_conda/data/Unet_Valid_Hair_Masks.npy", valid_masks
)

print("Finish save the data")

# TrainFile = path + "segmentation/train.txt"
# validateFile = path + "segmentation/val.txt"

# # train Data
# df = pd.read_csv(TrainFile, sep=" ", header=None)
# filesList = df[0].values

# # print(filesList)

# # load one image and one mask

# # image
# img = cv2.imread(imagesPath + "/0.jpg", cv2.IMREAD_COLOR)
# img = cv2.resize(img, (Width, Height))
# # cv2.imshow("img", img)

# # cv2.waitKey(0)


# # mask
# mask = cv2.imread(maskPath + "/0.jpg", cv2.IMREAD_GRAYSCALE)
# mask = cv2.resize(mask, (Width, Height))
# # cv2.imshow("mask", mask)

# # cv2.waitKey(0)

# # load all the train images and masks
# # ==================================


# print("Start loading the train images and masks ..............................")
# for file in filesList:
#     filePathForImage = imagesPath + f"/{file}" + ".jpg"
#     filePathForMask = maskPath + f"/{file}" + ".jpg"

#     # print(file)
#     img = Image.open(filePathForImage)
#     img = img.resize((Width, Height))
#     img = np.array(img)
#     img = img / 255.0  # normalize to [0, 1]
#     img = img.astype(np.float32)
#     allImages.append(img)
#     # img = cv2.imread(filePathForImage, cv2.IMREAD_COLOR)
#     # img = cv2.resize(img, (Width, Height))
#     # img = img / 255.0
#     # img = img.astype(np.float32)
#     if file == 10:
#         cv2.imshow("img", img)
#         cv2.waitKey(0)
#     # allImages.append(img)

#     # Load mask
#     mask = Image.open(filePathForMask).convert("L")  # 'L' mode represents grayscale
#     mask = mask.resize((Width, Height))
#     mask = np.array(mask)
#     maskImages.append(mask)

#     # mask = cv2.imread(filePathForMask, cv2.IMREAD_GRAYSCALE)
#     # mask = cv2.resize(mask, (Width, Height))
#     if file == 10:
#         cv2.imshow("img", mask)
#         cv2.waitKey(0)
#     # maskImages.append(mask)


# allImagesNP = np.array(allImages)
# maskImagesNP = np.array(maskImages)
# maskImagesNP = maskImagesNP.astype(int)  # convert the values to integers


# print("Shapes of train images and masks :")
# print(allImagesNP.shape)
# print(maskImagesNP.shape)
# print(maskImagesNP.dtype)

# # # load the Validate images and masks
# # # ==================================
# df = pd.read_csv(validateFile, sep=" ", header=None)
# filesList = df[0].values
# # print(f"file list {filesList}")

# print("Start loading the Validate images and masks ..............................")
# for file in filesList:
#     filePathForImage = imagesPath + f"/{file}" + ".jpg"
#     filePathForMask = maskPath + f"/{file}" + ".jpg"

#     # print(file)
#     img = Image.open(filePathForImage)
#     img = img.resize((Width, Height))
#     img = np.array(img)
#     img = img / 255.0  # normalize to [0, 1]
#     img = img.astype(np.float32)
#     allValidateImages.append(img)

#     # img = cv2.imread(filePathForImage, cv2.IMREAD_COLOR)
#     # img = cv2.resize(img, (Width, Height))
#     # img = img / 255.0
#     # img = img.astype(np.float32)
#     if file == 940:
#         cv2.imshow("img", img)
#         cv2.waitKey(0)
#     # allValidateImages.append(img)

#     mask = Image.open(filePathForMask).convert("L")  # 'L' mode represents grayscale
#     mask = mask.resize((Width, Height))
#     mask = np.array(mask)
#     maskValidatImages.append(mask)

#     # mask = cv2.imread(filePathForMask, cv2.IMREAD_GRAYSCALE)
#     # mask = cv2.resize(mask, (Width, Height))
#     if file == 940:
#         cv2.imshow("img", mask)
#         cv2.waitKey(0)
#     # maskValidatImages.append(mask)


# allValidateImagesNP = np.array(allValidateImages)
# maskValidateImagesNP = np.array(maskValidatImages)
# maskValidateImagesNP = maskValidateImagesNP.astype(
#     int
# )  # convert the values to integers


# print("Shapes of train images and masks :")
# print(allValidateImagesNP.shape)
# print(maskValidateImagesNP.shape)
# print(maskValidateImagesNP.dtype)

# print("Save the Data ......")

# np.save(
#     "C:/Users/Tarik/Desktop/unetv3/data/Unet-Hair-Train-Images.npy",
#     allImagesNP,
# )
# np.save(
#     "C:/Users/Tarik/Desktop/unetv3/data/Unet-Hair-Train-masks.npy",
#     maskImagesNP,
# )
# np.save(
#     "C:/Users/Tarik/Desktop/unetv3/data/Unet-Hair-Validate-Images.npy",
#     allValidateImagesNP,
# )
# np.save(
#     "C:/Users/Tarik/Desktop/unetv3/data/Unet-Hair-Validate-Masks.npy",
#     maskValidateImagesNP,
# )

# print("Finish save the data .............")
