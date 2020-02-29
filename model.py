import csv
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Convolution2D, Flatten, Dense, Dropout


# From one video, we can flip all the frames and steering angle numbers to generate
# "right handed" turning data and give the network more examples to train off
# initialize containers below
images = []
measurements = []
flipped_images = []
flipped_measurements = []

# Read in log file from simulator
with open('../sim-training-data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        imgpath = line[0].split('/', 3)[3] # culls the "home/workspace" part of the path off
        print("Loading frame:", imgpath)

        img = mpimg.imread('../' + imgpath)
        meas = float(line[3])

        # flip images and measurements here before appending to list
        images.append(img)
        flipped_images.append(np.fliplr(img))
        measurements.append(meas)
        flipped_measurements.append(-meas)

# Collect both normal and mirrored data into one container (respectively)
images.extend(flipped_images)
measurements.extend(flipped_measurements)

# Double check that images load properly
# print(images[0])
# plt.imshow(images[0])
# plt.show()

# Convert image/measurement containers to Numpy objects, Keras needs it
X_train = np.array(images)
y_train = np.array(measurements)

# Check the shape of these containers to ensure conversion worked
assert(X_train.shape[0] == y_train.shape[0])
print("Number of samples:", y_train.shape[0])
print("Image dimensions:", X_train[0].shape)


# Begin constructing network model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=X_train[0].shape))
model.add(Cropping2D(cropping=((65,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))                     # First two dense layers get dropout
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))                     # Dropout reduces overfitting significantly
model.add(Dense(50, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

# Mean Squared Error will work for the "how far off" assessment
# Adam optimizer works well with zero effort
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5, verbose=2)

# Save model after training
model.save('model.h5')
print("Model saved")

