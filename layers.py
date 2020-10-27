import coremltools

spec = coremltools.utils.load_spec("DeepLabV3_v4.mlmodel")

print(len(spec.neuralNetwork.layers))

for layer in spec.neuralNetwork.layers:
	if len(layer.resizeBilinear.targetSize) == 2:
	# 	if layer.resizeBilinear.targetSize[0] == 513:
	# 		print("jaja")
		print(layer.name)
		print(layer.resizeBilinear.targetSize)
		# layer.resizeBilinear.targetSize[0] = 1024
		# layer.resizeBilinear.targetSize[1] = 1024

	if layer.name == "concat:0" or layer.name == "ResizeBilinear_1:0" or layer.name == "aspp0/Relu:0":
		print(layer)

# coremltools.utils.save_spec(spec, 'DeepLabV3_v4.mlmodel')

	# if hasattr(layer, 'resizeBilinear'):
	# 	print(layer.resizeBilinear)
	# else:
	# 	print("No.")