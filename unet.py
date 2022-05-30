from keras.models import *
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from data import *


class myUnet(object):
	def __init__(self, img_rows = 512, img_cols = 512):
		self.img_rows = img_rows
		self.img_cols = img_cols
# 参数初始化定义
	def load_data(self):
		mydata = dataProcess(self.img_rows, self.img_cols)
		imgs_train, imgs_mask_train = mydata.load_train_data()
		imgs_test = mydata.load_test_data()
		return imgs_train, imgs_mask_train, imgs_test
# 载入数据
	def get_unet(self):
		inputs = Input((self.img_rows, self.img_cols,1))

		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		print ("conv1 shape:",conv1.shape)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		print ("conv1 shape:",conv1.shape)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		print ("pool1 shape:",pool1.shape)

		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		print ("conv2 shape:",conv2.shape)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		print ("conv2 shape:",conv2.shape)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		print ("pool2 shape:",pool2.shape)

		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		print ("conv3 shape:",conv3.shape)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		print ("conv3 shape:",conv3.shape)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		print ("pool3 shape:",pool3.shape)

		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		drop5 = Dropout(0.5)(conv5)

		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		merge6 = Concatenate(axis=3)([drop4, up6])
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		merge7 = Concatenate(axis=3)([conv3, up7])
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		merge8 = Concatenate(axis=3)([conv2,up8])
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		merge9 = Concatenate(axis=3)([conv1,up9])
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

		conv10 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
		print("conv10 shape:", conv10.shape)
		conv10 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
		print("conv10 shape:", conv10.shape)
		pool10 = MaxPooling2D(pool_size=(2, 2))(conv10)
		print("pool10 shape:", pool10.shape)

		conv11 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool10)
		print("conv11 shape:", conv11.shape)
		conv11 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
		print("conv11 shape:", conv11.shape)
		pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
		print("pool11 shape:", pool11.shape)

		conv12 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool11)
		print("conv12 shape:", conv12.shape)
		conv12 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv12)
		print("conv12 shape:", conv12.shape)
		pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)
		print("pool12 shape:", pool12.shape)

		conv13 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool12)
		conv13 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv13)
		drop13 = Dropout(0.5)(conv13)
		pool13 = MaxPooling2D(pool_size=(2, 2))(drop13)

		conv14 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool13)
		conv14 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv14)
		drop14 = Dropout(0.5)(conv14)

		up15 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
			UpSampling2D(size=(2, 2))(drop14))
		merge15 = Concatenate(axis=3)([drop13, up15])
		conv15 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge15)
		conv15 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv15)

		up16 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
			UpSampling2D(size=(2, 2))(conv15))
		merge16 = Concatenate(axis=3)([conv12, up16])
		conv16 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge16)
		conv16 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv16)

		up17 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
			UpSampling2D(size=(2, 2))(conv16))
		merge17 = Concatenate(axis=3)([conv11, up17])
		conv17 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge17)
		conv17 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv17)

		up18 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
			UpSampling2D(size=(2, 2))(conv17))
		merge18 = Concatenate(axis=3)([conv10, up18])
		conv18 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge18)
		conv18 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv18)
		conv18 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv18)
		conv19 = Conv2D(1, 1, activation='sigmoid')(conv18)

		model = Model(inputs, conv19)
		model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
		return model


	def train(self):
		print("loading data")
		imgs_train, imgs_mask_train, imgs_test = self.load_data()
		print("loading data done")
		model = self.get_unet()
		print("got unet")
		model_checkpoint = ModelCheckpoint('my_unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		model.fit(imgs_train, imgs_mask_train, batch_size=2, epochs=20, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

		print('predict test data')
		imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
		np.save('./results/imgs_mask_test.npy', imgs_mask_test)

	def save_img(self):
		print("array to image")
		imgs = np.load('./results/imgs_mask_test.npy')
		imgs_name = sorted(glob.glob("./raw/test"+"/*."+"tif"))
		for i in range(imgs.shape[0]):
			img = imgs[i]
			imgname = imgs_name[i]
			midname = imgname[imgname.rindex("/") + 1:]
			img_order = midname[:-4]
			img = array_to_img(img)
			img.save("./results/%s.jpg"%(img_order))

if __name__ == '__main__':
	myunet = myUnet()
	myunet.train()

	