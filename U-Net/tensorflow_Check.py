import tensorflow as tf

# TensorFlow'un hangi cihazı kullandığını gösterir
# to make sure tensorflow using the gpu for training
print("TensorFlow cihazı:", tf.test.gpu_device_name())
