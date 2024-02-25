# Recognizing digits from user's drawing

## Models:
- mlp_baseline: Basic MLP trained on MNIST dataset with 91% test accuracy. However, this model failed miserably when trying to predict my own handwriting. I ensured no data leakage during training, so the problem is probably that the MNIST dataset is too clean and different from real-life handwriting.

- cnn_baseline: Basic CNN trained on MNIST dataset with 94% test accuracy. Same problem as the MLP model.
