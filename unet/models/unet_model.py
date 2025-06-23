"""
U-Net model implementation for hair segmentation.
Provides a clean and modular implementation of the U-Net architecture.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, 
    MaxPool2D, UpSampling2D, Concatenate
)
from tensorflow.keras.models import Model
from typing import Tuple, List

from config import MODEL_CONFIG


class UNetModel:
    """
    U-Net model for semantic segmentation.
    
    Attributes:
        input_shape: Shape of input images (height, width, channels)
        num_filters: List of filter counts for each encoder/decoder level
        bridge_filters: Number of filters in the bridge layer
        output_channels: Number of output channels
        activation: Activation function for output layer
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = MODEL_CONFIG["input_shape"],
                 num_filters: List[int] = MODEL_CONFIG["num_filters"],
                 bridge_filters: int = MODEL_CONFIG["bridge_filters"],
                 output_channels: int = MODEL_CONFIG["output_channels"],
                 activation: str = MODEL_CONFIG["activation"]):
        """
        Initialize U-Net model with specified parameters.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_filters: List of filter counts for each encoder/decoder level
            bridge_filters: Number of filters in the bridge layer
            output_channels: Number of output channels
            activation: Activation function for output layer
        """
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.bridge_filters = bridge_filters
        self.output_channels = output_channels
        self.activation = activation
        self.model = None
        
    def _convolutional_block(self, x: tf.Tensor, num_filters: int) -> tf.Tensor:
        """
        Create a convolutional block with two consecutive conv layers.
        
        Args:
            x: Input tensor
            num_filters: Number of filters for convolution layers
            
        Returns:
            Output tensor after convolutional block
        """
        x = Conv2D(num_filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = Conv2D(num_filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        return x
    
    def build(self) -> Model:
        """
        Build the U-Net model architecture.
        
        Returns:
            Compiled U-Net model
        """
        # Input layer
        inputs = Input(self.input_shape)
        
        # Encoder path
        skip_connections = []
        x = inputs
        
        # Downsampling path
        for filters in self.num_filters:
            x = self._convolutional_block(x, filters)
            skip_connections.append(x)
            x = MaxPool2D((2, 2))(x)
        
        # Bridge
        x = self._convolutional_block(x, self.bridge_filters)
        
        # Decoder path
        # Reverse filters and skip connections for upsampling
        reversed_filters = list(reversed(self.num_filters))
        reversed_skip_connections = list(reversed(skip_connections))
        
        # Upsampling path
        for i, filters in enumerate(reversed_filters):
            x = UpSampling2D((2, 2))(x)
            skip_connection = reversed_skip_connections[i]
            x = Concatenate()([x, skip_connection])
            x = self._convolutional_block(x, filters)
        
        # Output layer
        x = Conv2D(self.output_channels, (1, 1), padding="same")(x)
        x = Activation(self.activation)(x)
        
        self.model = Model(inputs, x)
        return self.model
    
    def compile_model(self, 
                     optimizer: str = "adam",
                     loss: str = "binary_crossentropy",
                     metrics: List[str] = None) -> Model:
        """
        Compile the U-Net model.
        
        Args:
            optimizer: Optimizer for training
            loss: Loss function
            metrics: List of metrics to track
            
        Returns:
            Compiled model
        """
        if metrics is None:
            metrics = ["accuracy"]
            
        if self.model is None:
            self.build()
            
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return self.model
    
    def summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            self.build()
        self.model.summary()


def create_unet_model(input_shape: Tuple[int, int, int] = None,
                     **kwargs) -> UNetModel:
    """
    Factory function to create a U-Net model.
    
    Args:
        input_shape: Shape of input images
        **kwargs: Additional arguments for UNetModel
        
    Returns:
        UNetModel instance
    """
    if input_shape is None:
        input_shape = MODEL_CONFIG["input_shape"]
        
    return UNetModel(input_shape=input_shape, **kwargs)
