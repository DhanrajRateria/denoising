import tensorflow as tf
from tensorflow.keras import layers, Model

class UNetDenoiser:
    """U-Net architecture for image denoising."""
    
    def __init__(self, input_shape=(None, None, 1)):
        self.model = self._build_unet(input_shape)
        
    def _conv_block(self, inputs, filters, kernel_size=3):
        """Convolutional block with two conv layers."""
        x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(inputs)
        x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
        return x
    
    def _build_unet(self, input_shape):
        """Build U-Net architecture."""
        inputs = layers.Input(shape=input_shape)
        
        # Encoder
        conv1 = self._conv_block(inputs, 64)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = self._conv_block(pool1, 128)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = self._conv_block(pool2, 256)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        
        # Bridge
        conv4 = self._conv_block(pool3, 512)
        
        # Decoder
        up5 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv4)
        concat5 = layers.Concatenate()([conv3, up5])
        conv5 = self._conv_block(concat5, 256)
        
        up6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
        concat6 = layers.Concatenate()([conv2, up6])
        conv6 = self._conv_block(concat6, 128)
        
        up7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
        concat7 = layers.Concatenate()([conv1, up7])
        conv7 = self._conv_block(concat7, 64)
        
        # Output
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv7)
        
        return Model(inputs=inputs, outputs=outputs)
    
    def load_weights(self, weights_path):
        """Load pre-trained weights."""
        self.model.load_weights(weights_path)
    
    def denoise(self, image):
        """Denoise an image using the model."""
        # Ensure image is in correct format (add batch dimension if needed)
        if len(image.shape) == 2:
            image = image[..., None]
        if len(image.shape) == 3:
            image = image[None, ...]
            
        # Normalize image to [0, 1]
        image = image.astype('float32') / 255.0
        
        # Predict
        denoised = self.model.predict(image)
        
        # Convert back to uint8
        denoised = (denoised[0] * 255).astype('uint8')
        return denoised