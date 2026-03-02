#include "matrix.h"
#include "model.h"
#include <stdio.h>
#include <math.h>
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

void create_mnist_model(model_context *model) {
  model_var *input = mv_create(model, 784, 1, MV_FLAG_INPUT);
  model_var *w0 =
      mv_create(model, 16, 784, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  model_var *w1 =
      mv_create(model, 16, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  model_var *w2 =
      mv_create(model, 10, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

  float bound0 = sqrtf(6.0f / (784 + 16));
  float bound1 = sqrtf(6.0f / (16 + 16));
  float bound2 = sqrtf(6.0f / (16 + 10));

  mat_fill_rand(w0->val, -bound0, bound0);
  mat_fill_rand(w1->val, -bound1, bound1);
  mat_fill_rand(w2->val, -bound2, bound2);

  model_var *b0 =
      mv_create(model, 16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  model_var *b1 =
      mv_create(model, 16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  model_var *b2 =
      mv_create(model, 10, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

  model_var *z0_a = mv_matmul(model, w0, input, 0);
  model_var *z0_b = mv_add(model, z0_a, b0, 0);
  model_var *a0 = mv_relu(model, z0_b, 0);

  model_var *z1_a = mv_matmul(model, w1, a0, 0);
  model_var *z1_b = mv_add(model, z1_a, b1, 0);
  model_var *z1_c = mv_relu(model, z1_b, 0);
  model_var *a1 = mv_add(model, z1_c, a0, 0);

  model_var *z2_a = mv_matmul(model, w2, a1, 0);
  model_var *z2_b = mv_add(model, z2_a, b2, 0);
  model_var *output = mv_softmax(model, z2_b, MV_FLAG_OUTPUT);

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
  create_mnist_model(model);
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

                                       .epochs = 5,
                                       .batch_size = 20,
                                       .learning_rate = 0.005};
  model_train(model, &training_desc);

  memcpy(model->input->val->data, test_images->data + (2 * 784), sizeof(float) * 784);

  model_feedforward(model);

  printf("post training output: \n");

  _draw_mnist_digit(test_images->data + 2 * 784);
  for (unsigned int i = 0; i < 10; i++) {
    printf("%f ", model->output->val->data[i]);
  }
  printf("\n");
  return 0;
}
