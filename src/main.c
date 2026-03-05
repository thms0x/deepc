#include "matrix.h"
#include "model.h"
#include <layer.h>
#include <stdio.h>
#include <string.h>

void _draw_mnist_digit(float *data) {
  for (int i = 0; i < 28; i++) {
    for (int j = 0; j < 28; j++) {
      float num = data[i * 28 + j];
      int col = 232 + (int)(num * 23);
      printf("\x1b[48;5;%dm ", col);
    }
    printf("\n");
  }
  printf("\x1b[0m");
}

void create_mnist_model_test(model_context *model) {
  model_var *input = mv_create(model, 784, 1, MV_FLAG_INPUT);
  model_var *z0 = layer_fully_connected(model, input, 784, 128);
  model_var *a0 = mv_relu(model, z0, 0);
  model_var *z1 = layer_fully_connected(model, a0, 128, 64);
  model_var *a1 = mv_relu(model, z1, 0);
  model_var *z2 = layer_fully_connected(model, a1, 64, 10);
  model_var *output = mv_softmax(model, z2, MV_FLAG_OUTPUT);
  model_var *y = mv_create(model, 10, 1, MV_FLAG_DESIRED_OUPUT);
  model_var *cost = mv_cross_entropy(model, output, y, MV_FLAG_COST);
}

int main(void) {
  matrix *train_images = mat_load(60000, 784, "data/mnist_train_images.mat");
  matrix *test_images = mat_load(10000, 784, "data/mnist_test_images.mat");
  matrix *train_labels = mat_create(60000, 10);
  matrix *test_labels = mat_create(10000, 10);

  {
    matrix *train_labels_file =
        mat_load(60000, 1, "data/mnist_train_labels.mat");
    matrix *test_labels_file = mat_load(10000, 1, "data/mnist_test_labels.mat");

    for (int i = 0; i < 60000; i++) {
      int num = train_labels_file->data[i];
      train_labels->data[i * 10 + num] = 1.0f;
    }
    for (int i = 0; i < 10000; i++) {
      int num = test_labels_file->data[i];
      test_labels->data[i * 10 + num] = 1.0f;
    }
  }

  for (int i = 0; i < 10; i++)
    printf("%.0f", train_labels->data[2 * 10 + i]);

  model_context *model = model_create();
  create_mnist_model_test(model);
  model_compile(model);

  memcpy(model->input->val->data, train_images->data, sizeof(float) * 784);
  model_feedforward(model);

  _draw_mnist_digit(train_images->data + 0 * 784);
  printf("pre training output: \n");
  for (unsigned int i = 0; i < 10; i++) {
    printf("%f ", model->output->val->data[i]);
  }
  printf("\n");

  model_training_desc training_desc = {.train_images = train_images,
                                       .train_labels = train_labels,
                                       .test_images = test_images,
                                       .test_labels = test_labels,

                                       .epochs = 1,
                                       .batch_size = 20,
                                       .learning_rate = 0.01};
  model_train(model, &training_desc);
  int right_predictions = 0;
  int wrong_predictions = 0;
  int accuracy = 0;
  printf("post training output: \n");

  for (unsigned int i = 0; i < 10000; i++) {
    memcpy(model->input->val->data, test_images->data + (i * 784),
           sizeof(float) * 784);

    model_feedforward(model);

    // _draw_mnist_digit(test_images->data + i * 784);
    if (test_labels->data[i * 10 + mat_argmax(model->output->val)] == 1.0f )
           right_predictions++;
    else{
      wrong_predictions++;
    }
  }
  printf("Right predictions: %d, Wrong predictions: %d, Accuracy: %.2f%%\n",
         right_predictions, wrong_predictions,
         (float)right_predictions / (float)(right_predictions + wrong_predictions) * 100.0f);

  mat_free(train_images);
  mat_free(test_images);
  mat_free(train_labels);
  mat_free(test_labels);
  return 0;
}
