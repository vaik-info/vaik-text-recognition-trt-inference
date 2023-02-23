# vaik-text-recognition-trt-inference

Inference by text recognition TensorRT model


## Install

``` shell
pip install git+https://github.com/vaik-info/vaik-text-recognition-trt-inference.git
```

## Usage

### Example
```python
import numpy as np
from PIL import Image
from vaik_text_recognition_trt_inference.trt_model import TrtModel

classes = TrtModel.char_json_read('/home/kentaro/Github/vaik-text-recognition-trt-trainer/data/jpn_character.json')
model_path = '/home/kentaro/.vaik_text_recognition_pb_exporter/model.trt'
model = TrtModel(model_path, classes)

image1 = np.asarray(Image.open("/home/kentaro/Desktop/images/いわき_0333.png").convert('RGB'))

output, raw_pred = model.inference([image1])
```


#### Output

- output

```text
[{'text': 'いわき', 'classes': [113, 155, 118], 'scores': 0.9999999991409891}]
```