# Import required modules

import numpy as np

import PIL.Image

import matplotlib.pyplot as plt
import matplotlib as mpl

import tensorflow as tf

# Define the data loader
class StyleTransferDataloader():

    def __init__(self, img_path, max_dim = 512):
        self.max_dim = max_dim

    # Define utility functions
    def load_image(self, path_to_img):
        '''Preprocess and load image.'''
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = self.max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]

        return img

    def display_image(self, image, title=None):
      '''Display image.'''
      if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
      plt.imshow(image)
      if title:
        plt.title(title)


    def tensor_to_image(self, tensor):
        '''Return an image for a given tensor.'''
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]

        return PIL.Image.fromarray(tensor)

# Define the model
class StyleTransferModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers, trainable=False, optimizer_choice='Adam', learning_rate=0.02, beta_1=0.99, epsilon=1e-1, content_weight=1e4, style_weight=1e-2, variation_weight=30):
        super(StyleTransferModel, self).__init__()

        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(self.style_layers)
        self.num_content_layers = len(self.content_layers)
        self.vgg = self.set_vgg_layers(layer_names = self.style_layers + self.content_layers)
        self.trainable = trainable

		# Hyperparameters
		self.learning_rate = learning_rate
		self.beta_1 = beta_1
		self.epsilon = epsilon

		# Hyperparameters - weights of content loss, style loss and total variation loss
		self.content_weight=content_weight
		self.style_weight=style_weight
		self.variation_weight=variation_weight

		# Optimizer used
		if optimizer_choice.lower() = 'adam':
			self.opt = tf.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, epsilon=self.epsilon)
		else:
			print('\nPlease choose another optimizer...\nPossible choices include-\n 1. Adam')

    # Base model
    def set_vgg_layers(self, inputs):
        '''Creates a vgg model that returns a list of intermediate output values.'''
        layer_names = inputs
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = self.trainable

        outputs = [vgg.get_layer(name).output for name in layer_names]
        model = tf.keras.Model([vgg.input], outputs)

        return model

    def gram_matrix(self, inputs):
	    '''Computes gram matrix.'''
	    input_tensor = inputs
	    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
	    input_shape = tf.shape(input_tensor)
	    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

	    return result / (num_locations)

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)

        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])

        style_outputs = [self.gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

    @tf.function()
	def total_loss(self, inputs):
		'''Compute total loss inclusive of style loss, content loss and variation loss.'''
		image = inputs
		outputs = self(image)

		# Style loss
		style_targets = self(style_image)['style']
		style_outputs = outputs['style']
		style_loss = tf.add_n([tf.reduce_mean((style_targets[name]-style_outputs[name])**2) for name in style_outputs.keys()])
		style_loss *= style_weight / self.num_style_layers

		# Content loss
		content_targets = self(content_image)['content']
		content_outputs = outputs['content']
		content_loss = tf.add_n([tf.reduce_mean((content_targets[name]-content_outputs[name])**2) for name in content_outputs.keys()])
		content_loss *= content_weight / self.num_content_layers

		# Variation loss
		variation_loss = variation_weight*tf.image.total_variation(image)

		# Total loss
		total_loss = style_loss + content_loss + variation_loss

  		return [total_loss, content_loss, style_loss, variation_loss]

	@tf.function()
	def train_step(self, inputs):
		'''Train model.'''
		image = inputs

	    with tf.GradientTape() as tape:
	    	loss = self.total_loss(image)
	    	total_loss = loss[0]
	    grad = tape.gradient(total_loss, image)
	    self.opt.apply_gradients([(grad, image)])
	    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

		return loss


# Input config
data_dir = ''
content_path = data_dir + 'content-image.jpg'
style_path = data_dir + 'style-image.jpg'
stylized_image_path = data_dir + 'stylized-image.jpg'

epochs = 10
steps_per_epoch = 100

# View inputs
# Data loader
style_transfer_dataloader = StyleTransferDataloader()

# Content image
content_image = style_transfer_dataloader.load_image(path_to_img=content_path)
plt.subplot(1, 2, 1)
style_transfer_dataloader.display_image(image=content_image, title='Content Image')

# Style image
style_image = style_transfer_dataloader.load_image(path_to_img=style_path)
plt.subplot(1, 2, 2)
style_transfer_dataloader.display_image(image=style_image, title='Style Image')

# Create model
# Required content and style layers
content_layers = ['block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

# Build model
extractor = StyleTransferModel(style_layers, content_layers)

# Training
image = tf.Variable(content_image)

step = 0
for n in range(epochs):
	progBar = tf.keras.utils.Progbar(steps_per_epoch, stateful_metrics=['loss_fn'], verbose=1)
  	for m in range(steps_per_epoch):
	    loss = train_step(image)

	    total_loss = loss[0].numpy()
	    content_loss = loss[1].numpy()
	    style_loss= loss[2].numpy()
	    variation_loss = loss[3].numpy()

        values = [('total loss', total_loss), ('content loss', content_loss), ('style loss', style_loss), ('variation loss', variation_loss)]
        progBar.add(1, values)

        step += 1


# Save stylized image
style_transfer_dataloader.tensor_to_image(image).save(stylized_image_path)

# View stylized image
stylized_image = style_transfer_dataloader.load_image(path_to_img=stylized_image_path)
style_transfer_dataloader.display_image(image=stylized_image, title='Stylized Image')