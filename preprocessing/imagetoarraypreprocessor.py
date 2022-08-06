from keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:
    """
    The benefit of defining a class to handle this type of image preprocessing rather than simply
    calling img_to_array on every single image is that we can now chain preprocessors together as
    we load datasets from disk.
    """
    
    def __init__(self, dataFormat=None):  # dataFormat defaults to None, which indicates that the setting inside keras.json should be used.
        # store the image data format
        self.dataFormat = dataFormat
        
    def preprocess(self, image):
        # apply the keras utility function that correctly rearranges the dimension of the image
        return img_to_array(image, data_format=self.dataFormat)
    

    
