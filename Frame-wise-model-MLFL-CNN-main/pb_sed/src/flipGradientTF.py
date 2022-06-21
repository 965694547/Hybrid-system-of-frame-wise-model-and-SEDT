import tensorflow as tf
from keras.engine import Layer
import keras.backend as K

def reverse_gradient(X, op_name, hp_lambda):
	'''Flips the sign of the incoming gradient during training.'''
	try:
		reverse_gradient.num_calls += 1
	except AttributeError:
		reverse_gradient.num_calls = 1

	grad_name = "GradientReversal_%s_%d" % (op_name, reverse_gradient.num_calls)
	@tf.RegisterGradient(grad_name)
	def _flip_gradients(op, grad):
		print(grad_name)
		if hp_lambda == 0:
			return [tf.zeros_like(grad)]
		if hp_lambda > 0:
			return [grad * hp_lambda]  
		if hp_lambda < 0:
			return [(-hp_lambda) * tf.negative(grad)]

	g = K.get_session().graph
	with g.gradient_override_map({'Identity': grad_name}):
		y = tf.identity(X)

	return y

class GradientReversal(Layer):
	'''Flip the sign of gradient during training.'''
	def __init__(self, op_name, hp_lambda, **kwargs):
		super(GradientReversal, self).__init__(**kwargs)
		self.supports_masking = False
		self.hp_lambda = hp_lambda
		self.op_name = op_name

	def build(self, input_shape):
		self.trainable_weights = []

	def call(self, x, mask=None):
		return reverse_gradient(x, self.op_name, self.hp_lambda)

	def get_output_shape_for(self, input_shape):
		return input_shape

	def get_config(self):
		config = {'hp_lambda': self.hp_lambda, 'grl' : self.grl, 'op_name': self.op_name}
		base_config = super(GradientReversal, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
