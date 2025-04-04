import pandas as pd
from mtcnn import MTCNN
from mtcnn.utils.plotting import plot
import matplotlib.pyplot as plt
from mtcnn.utils.images import load_image
from sklearn.model_selection import train_test_split
from load import Load

detector = MTCNN(device='cpu')

anger_file_path = '/Users/suryapasupuleti/Downloads/6 Emotions for image classification/anger'

load = Load()

angry_imgs = load.load_data(anger_file_path)

df = pd.DataFrame(angry_imgs)

X = df.iloc[: len(df) // 2]
Y = df.iloc[len(df) // 2:]


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)













'''

image = load_image('/Users/suryapasupuleti/Downloads/6 Emotions for image classification/anger/-win-holding-his-fists-shout-wow-mature-hispanic-man-happy-his-win-122652456.jpg')

res = detector.detect_faces(image)

plt.imshow(plot(image, res))
plt.show()

'''






