# Emotion Recognition : A Comparitive Analysis
This project utilizes Deep Learning models such as ResNet-50, MobileNet, EfficientNet and a custom CNN for emotion detection. The CNN model gave maximum accuracy.
# Data Preprocessing
The [Face expression recognition dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) dataset from Kaggle consists of images for seven emotion classes: sad, happy, angry, disgust, fear, neutral, and surprise. The dataset comprises a total of 28,820 images for training and 7,066 images for testing, with each image having dimensions of 48x48 pixels. An imbalance was observed in the dataset, which means that certain emotion classes had more data samples than others. To address this issue and enhance the model's performance, data augmentation techniques were employed during the preprocessing stage. These techniques involve generating additional training examples by applying transformations like rotation, scaling, and flipping to the existing images, thereby helping to mitigate the data imbalance and improve the model's ability to generalize across all emotion classes.
# Models
In this project, four different models were utilized for the task of emotion recognition from facial expressions: Custom CNN, EfficientNet, ResNet50, and MobileNet. Each of these models played a crucial role in exploring the effectiveness of various deep learning architectures for this specific application.

1. **Custom CNN (Convolutional Neural Network):**
   - The custom CNN architecture was designed from scratch, tailored to the task of facial emotion recognition. It featured a series of convolutional layers followed by pooling and fully connected layers. This model was trained to capture the unique features and patterns associated with different facial expressions.

2. **EfficientNet:**
   - EfficientNet is a powerful deep learning model known for its efficiency and accuracy. It was employed in this project to leverage its pre-trained weights, which had been learned from extensive datasets like ImageNet. Fine-tuning was applied to adapt the model for facial emotion recognition, and it played a pivotal role in achieving high accuracy.

3. **ResNet50:**
   - ResNet50 is a widely recognized architecture famous for its residual connections. In this project, it was selected for its strong feature extraction capabilities. By fine-tuning ResNet50 on the facial emotion dataset, it allowed the model to grasp intricate details and nuances present in the images, contributing to accurate emotion recognition.

4. **MobileNet:**
   - MobileNet is known for its efficiency and lightweight design, making it suitable for resource-constrained environments. It was another model choice to explore how well a lighter architecture can perform on the task. With fine-tuning, MobileNet showcased its ability to efficiently recognize emotions from facial expressions.

Each of these models was carefully considered to offer a diverse range of architectures, from custom designs to pre-trained state-of-the-art networks. By using and comparing these models, the project aimed to identify which architecture would provide the most accurate and efficient solution for facial emotion recognition, contributing to a comprehensive understanding of their performance in this context.
# Evaluation
To conduct a robust evaluation, a test set consisting of 30 random images was selected. These images were distinct from the training data, ensuring an unbiased assessment of the models' generalization capabilities. Each model, including the Custom CNN, EfficientNet, ResNet50, and MobileNet, was subjected to this evaluation. Here are the reults.
1. **Custom CNN (Convolutional Neural Network):**
   - ![image](https://github.com/chetan0220/emotion_detection/assets/97821311/be548efc-41e2-40d5-a91e-97d5a9bc8539)

2. **EfficientNet:**
   - ![image](https://github.com/chetan0220/emotion_detection/assets/97821311/80068b4d-5771-4695-8152-6a91f0c6c4c8)

3. **ResNet50:**
   ![image](https://github.com/chetan0220/emotion_detection/assets/97821311/d489866d-3a2b-4e29-b25e-c79658fc780a)

4. **MobileNet:**
   ![image](https://github.com/chetan0220/emotion_detection/assets/97821311/ae5ea2bb-f4d1-4a8b-a39c-1a815fecf4b3)

# Results
| Model          | Accuracy | Loss    |
|----------------|----------|---------|
| Custom CNN    | 0.61     | 1.07    |
| EfficientNet  | 0.60     | 1.09    |
| ResNet50      | 0.51     | 1.37    |
| MobileNet     | 0.57     | 1.65    |

1. **Comparitive Accuracy Graph**<br>
    ![image](https://github.com/chetan0220/emotion_detection/assets/97821311/e4f7d640-c5e6-479b-98c4-fe35541f5961)<br>

2. **Comparitive Loss Graph**<br>
   ![image](https://github.com/chetan0220/emotion_detection/assets/97821311/4d1c998b-9c93-42f0-9582-e26ee3cc02e4)<br>

   
