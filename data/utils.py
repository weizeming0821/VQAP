from torchvision import transforms


"""构建 DINOv2 图像预处理流程。"""
def build_dinov2_transform(input_size: int = 224) -> transforms.Compose:
	if input_size <= 0:
		raise ValueError("input_size must be a positive integer")
	if input_size % 14 != 0:
		raise ValueError("input_size must be a multiple of 14 for DINOv2 models")

	return transforms.Compose(
		[
			transforms.Resize(input_size),
			transforms.CenterCrop(input_size),
			transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225],
			),
		]
	)
