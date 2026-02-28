#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int rows;
  int cols;
  float *data;
} matrix;

matrix *mat_create(int rows, int cols);
matrix *mat_load(int rows, int cols, const char *filename);

int mat_copy(matrix *dst, matrix *src);
void mat_clear(matrix *m);
void mat_scale(matrix *mat, float scale);
int mat_transpose(matrix *m);
void mat_sum(matrix *m);
int mat_add(matrix *out, const matrix *a, const matrix *b);
int mat_sub(matrix *out, const matrix *a, const matrix *b);
int mat_mul(matrix *out, const matrix *a, const matrix *b, int zero_out,
            int transpose_a, int transpose_b);
void mat_fill(matrix *m, float x);
int mat_relu(matrix *out, const matrix *in);
int mat_softmax(matrix *out, const matrix *in);
int mat_cross_entropy_loss(matrix *out, const matrix *predictions,
                           const matrix *labels);
int mat_relu_add_grad(matrix *out, const matrix *in);
int mat_softmax_add_grad(matrix *out, const matrix *softmax_out);
int mat_cross_entropy_loss_grad(matrix *out, const matrix *predictions,
                                const matrix *labels);

matrix *mat_create(int rows, int cols) {
  matrix *m = (matrix *)malloc(sizeof(matrix));
  m->rows = rows;
  m->cols = cols;
  m->data = (float *)calloc(rows * cols, sizeof(float));
  return m;
}

matrix *mat_load(int rows, int cols, const char *filename) {
  matrix *m = mat_create(rows, cols);
  FILE *f = fopen(filename, "rb");
  if (!f) {
    fprintf(stderr, "Failed to open file: %s\n", filename);
    free(m->data);
    free(m);
    return NULL;
  }

  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);

  if (size != rows * cols * sizeof(float)) {
    fprintf(stderr, "File size does not match expected matrix size\n");
    free(m->data);
    free(m);
    return NULL;
  }

  fread(m->data, sizeof(float), rows * cols, f);
  fclose(f);
  return m;
}

int mat_copy(matrix *dst, matrix *src) {
  if (dst->rows != src->rows || dst->cols != src->cols)
    return 0;
  memcpy(dst->data, src->data, sizeof(float) * dst->rows * dst->cols);
  return 1;
}

void mat_clear(matrix *m) {
  memset(m->data, 0, sizeof(float) * m->rows * m->cols);
}

void mat_fill(matrix *m, float x) {
  for (int i = 0; i < m->rows * m->cols; i++) {
    m->data[i] = x;
  }
}

void mat_scale(matrix *m, float scale) {
  for (int i = 0; i < m->rows * m->cols; i++) {
    m->data[i] *= scale;
  }
}

int mat_transpose(matrix *m) {
  matrix *t = mat_create(m->cols, m->rows);
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      t->data[j * t->cols + i] = m->data[i * m->cols + j];
    }
  }
  mat_copy(m, t);
  free(t->data);
  free(t);
  return 1;
}

int mat_add(matrix *out, const matrix *a, const matrix *b) {
  if (a->rows != b->rows || a->cols != b->cols)
    return 0;
  if (out->rows != a->rows || out->cols != a->cols)
    return 0;

  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      out->data[i * out->cols + j] =
          a->data[i * a->cols + j] + b->data[i * b->cols + j];
    }
  }
  return 1;
}

int mat_sub(matrix *out, const matrix *a, const matrix *b) {
  if (a->rows != b->rows || a->cols != b->cols)
    return 0;
  if (out->rows != a->rows || out->cols != a->cols)
    return 0;

  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      out->data[i * out->cols + j] =
          a->data[i * a->cols + j] - b->data[i * b->cols + j];
    }
  }
  return 1;
}

void _mat_mul_nn(matrix *out, const matrix *a, const matrix *b) {
  for (int i = 0; i < out->rows; i++) {
    for (int k = 0; k < a->cols; k++) {
      for (int j = 0; j < out->cols; j++) {
        out->data[i * out->cols + j] +=
            a->data[i * a->cols + k] * b->data[k * b->cols + j];
      }
    }
  }
}

void _mat_mul_nt(matrix *out, const matrix *a, const matrix *b) {
  for (int i = 0; i < out->rows; i++) {
    for (int j = 0; j < out->cols; j++) {
      for (int k = 0; k < a->cols; k++) {
        out->data[i * out->cols + j] +=
            a->data[i * a->cols + k] * b->data[j * b->cols + k];
      }
    }
  }
}
void _mat_mul_tn(matrix *out, const matrix *a, const matrix *b) {
  for (int k = 0; k < a->rows; k++) {
    for (int i = 0; i < out->rows; i++) {
      for (int j = 0; j < out->cols; j++) {
        out->data[i * out->cols + j] +=
            a->data[k * a->cols + i] * b->data[k * b->cols + j];
      }
    }
  }
}
void _mat_mul_tt(matrix *out, const matrix *a, const matrix *b) {
  for (int i = 0; i < out->rows; i++) {
    for (int j = 0; j < out->cols; j++) {
      for (int k = 0; k < a->cols; k++) {
        out->data[i * out->cols + j] +=
            a->data[k * a->cols + i] * b->data[j * b->cols + k];
      }
    }
  }
}
int mat_mul(matrix *out, const matrix *a, const matrix *b, int zero_out,
            int transpose_a, int transpose_b) {
  // optimized for best cache locality
  // TODO: reduce index arithmetic
  int a_rows = transpose_a ? a->cols : a->rows;
  int a_cols = transpose_a ? a->rows : a->cols;
  int b_rows = transpose_b ? b->cols : b->rows;
  int b_cols = transpose_b ? b->rows : b->cols;

  if (a_cols != b_rows)
    return 0;
  if (out->rows != a_rows || out->cols != b_cols)
    return 0;

  if (zero_out)
    mat_clear(out);
  int transpose = (transpose_a << 1) | transpose_b;
  switch (transpose) {
  case 0b00:
    _mat_mul_nn(out, a, b);
    break;
  case 0b01:
    _mat_mul_nt(out, a, b);
    break;
  case 0b10:
    _mat_mul_tn(out, a, b);
    break;
  case 0b11:
    _mat_mul_tt(out, a, b);
    break;
  }
  return 1;
}

int mat_relu(matrix *out, const matrix *in) {
  if (out->rows != in->rows || out->cols != in->cols)
    return 0;

  for (int i = 0; i < in->rows * in->cols; i++) {
    out->data[i] = in->data[i] > 0 ? in->data[i] : 0;
  }
  return 1;
}

int mat_softmax(matrix *out, const matrix *in) {
  if (out->rows != in->rows || out->cols != in->cols)
    return 0;

  float sum = 0;
  for (int i = 0; i < out->rows * out->cols; i++) {
    out->data[i] = expf(in->data[i]);
    sum += out->data[i];
  }
  mat_scale(out, 1.0f / sum);
  return 2;
}

int mat_cross_entropy_loss(matrix *out, const matrix *predictions,
                           const matrix *labels) {
  if (out->rows != 1 || out->cols != 1)
    return 0;
  if (predictions->rows != labels->rows || predictions->cols != labels->cols)
    return 0;
  for (int i = 0; i < out->rows * out->cols; i++) {
    out->data[i] = predictions->data[i] > 0
                       ? -labels->data[i] * logf(predictions->data[i])
                       : 0;
  }
  return 1;
}

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
    for (int i = 0; i < 1000; i++) {
      int num = test_labels_file->data[i];
      test_labels->data[i * 10 + num] = 1.0f;
    }
  }

  _draw_mnist_digit(train_images->data + 2 * 784);
  for (int i = 0; i < 10; i++)
    printf("%.0f", train_labels->data[2 * 10 + i]);
}
