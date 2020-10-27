# https://machinethink.net/blog/coreml-image-mlmultiarray/
import coremltools
import coremltools.proto.FeatureTypes_pb2 as ft
from coremltools.models.neural_network import flexible_shape_utils

spec = coremltools.utils.load_spec("DeepLabV3Int8LUT.mlmodel")
size = 1024

input = spec.description.input[0]
input.type.imageType.height = size
input.type.imageType.width = size

output = spec.description.output[0]
output.type.imageType.height = size
output.type.imageType.width = size
output.type.imageType.colorSpace = ft.ImageFeatureType.GRAYSCALE
# coremltools.utils.save_spec(spec, "DeepLabV3Int8Image_v2.mlmodel")

# print(spec.description)

img_size_ranges = flexible_shape_utils.NeuralNetworkImageSizeRange()
img_size_ranges.add_height_range((512, 1024))
img_size_ranges.add_width_range((512, 1024))
flexible_shape_utils.update_image_size_range(spec, feature_name='image', size_range=img_size_ranges)
flexible_shape_utils.update_image_size_range(spec, feature_name='semanticPredictions', size_range=img_size_ranges)
coremltools.utils.save_spec(spec, 'DeepLabV3Int8LUT_v3.mlmodel')