from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob


class dataProcess(object):
	def __init__(self, out_rows, out_cols, data_path = r"/Users/shiyufan/unet/raw/train", label_path = r"/Users/shiyufan/unet/raw/label",
				 test_path = r"/Users/shiyufan/unet/raw/test", npy_path = "./npydata", result_path = r"/Users/shiyufan/unet/results", img_type = "tif"):

		self.out_rows = out_rows
		self.out_cols = out_cols
		self.data_path = data_path
		self.label_path = label_path
		self.img_type = img_type
		self.test_path = test_path
		self.npy_path = npy_path
		self.result_path = result_path
		if not os.path.exists(self.npy_path):
			os.mkdir(self.npy_path)
		if not os.path.exists(self.result_path):
			os.mkdir(self.result_path)

	def create_train_data(self):
		i = 0
		print('-'*30)
		print('Creating training images...')
		print('-'*30)
		imgs = glob.glob(self.data_path+"/*."+self.img_type)
		print("Number of train images is {}".format(len(imgs)))

		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:]
			img = load_img(self.data_path + "/" + midname,grayscale = True)
			label = load_img(self.label_path + "/" + midname,grayscale = True)
			img = img_to_array(img)
			label = img_to_array(label)
			imgdatas[i] = img
			imglabels[i] = label
			if i % 100 == 0:
				print('Done: {0}/{1} images'.format(i, len(imgs)))
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_train', imgdatas)
		np.save(self.npy_path + '/imgs_mask_train', imglabels)
		print('Saving to imgs_train.npy and imgs_mask_train.npy files done.')

	def create_test_data(self):
		i = 0
		print('-'*30)
		print('Creating test images...')
		print('-'*30)
		imgs = glob.glob(self.test_path+"/*."+self.img_type)
		imgs = sorted(imgs)
		print("Number of test images is {}".format(len(imgs)))
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:]
			img = load_img(self.test_path + "/" + midname,grayscale = True)
			img = img_to_array(img)
			imgdatas[i] = img
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_test.npy', imgdatas)
		print('Saving to imgs_test.npy files done.')

	def load_train_data(self):
		print('-'*30)
		print('load train images...')
		print('-'*30)
		imgs_train = np.load(self.npy_path+"/imgs_train.npy")
		imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train.npy")
		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
		imgs_train /= 255
		mean = imgs_train.mean(axis = 0)
		imgs_train -= mean
		imgs_mask_train /= 255
		imgs_mask_train[imgs_mask_train > 0.5] = 1
		imgs_mask_train[imgs_mask_train <= 0.5] = 0
		return imgs_train,imgs_mask_train

	def load_test_data(self):
		print('-'*30)
		print('load test images...')
		print('-'*30)
		imgs_test = np.load(self.npy_path+"/imgs_test.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255
		mean = imgs_test.mean(axis = 0)
		imgs_test -= mean
		return imgs_test


if __name__ == "__main__":

	mydata = dataProcess(512,512)
	mydata.create_train_data()
	mydata.create_test_data()

	imgs_train,imgs_mask_train = mydata.load_train_data()
	imgs_test = mydata.load_test_data()
	print(imgs_train.shape,imgs_mask_train.shape, imgs_test.shape)
