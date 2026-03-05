#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

void mat_fill_rand(matrix *m, float lower, float upper) {
  srand(time(NULL));
  for (int i = 0; i < m->rows * m->cols; i++) {
    m->data[i] = ((float)rand() / (float)RAND_MAX) * (upper - lower) + lower;
  }
}

void mat_scale(matrix *m, float scale) {
  for (int i = 0; i < m->rows * m->cols; i++) {
    m->data[i] *= scale;
  }
}

float mat_sum(matrix *mat) {
  unsigned int size = (unsigned int)mat->rows * mat->cols;

  float sum = 0.0f;
  for (unsigned int i = 0; i < size; i++) {
    sum += mat->data[i];
  }

  return sum;
}

int mat_argmax(matrix *mat) {
  unsigned int size = (unsigned int)mat->rows * mat->cols;

  int max_i = 0;
  for (unsigned int i = 0; i < size; i++) {
    if (mat->data[i] > mat->data[max_i]) {
      max_i = i;
    }
  }

  return max_i;
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
  return 1;
}

int mat_cross_entropy(matrix *out, const matrix *predictions,
                      const matrix *labels) {
  if (predictions->rows != labels->rows || predictions->cols != labels->cols)
    return 0;
  if (out->rows != predictions->rows || out->cols != predictions->cols)
    return 0;
  for (int i = 0; i < out->rows * out->cols; i++) {
    out->data[i] = predictions->data[i] > 0
                       ? -labels->data[i] * logf(predictions->data[i])
                       : 0;
  }
  return 1;
}

int mat_relu_add_grad(matrix *out, const matrix *in, const matrix *grad) {
  if (out->rows != in->rows || out->cols != in->cols)
    return 0;

  if (out->rows != grad->rows || out->cols != grad->cols)
    return 0;

  for (int i = 0; i < in->rows * in->cols; i++) {
    out->data[i] += in->data[i] > 0.0f ? grad->data[i] : 0.0f;
  }
  return 1;
}
int mat_softmax_add_grad(matrix *out, const matrix *softmax_out,
                         const matrix *grad) {

  int size = out->rows * out->cols;

  float dot = 0.0f;
  for (int j = 0; j < size; j++) {
    dot += softmax_out->data[j] * grad->data[j];
  }

  for (int i = 0; i < size; i++) {
    float s_i = softmax_out->data[i];
    float g_i = grad->data[i];
    out->data[i] += s_i * (g_i - dot);
  }

  return 1;
}

int mat_cross_entropy_add_grad(matrix *p_grad, matrix *q_grad,
                               const matrix *predictions, const matrix *labels,
                               const matrix *grad) {
  if (predictions->rows != labels->rows || predictions->cols != labels->cols)
    return 0;

  unsigned int size = (unsigned int)predictions->rows * predictions->cols;

  if (p_grad != NULL) {
    if (p_grad->rows != predictions->rows ||
        p_grad->cols != predictions->cols) {
      return 0;
    }
    for (unsigned int i = 0; i < size; i++) {
      p_grad->data[i] +=
          -(labels->data[i] / (predictions->data[i] + 1e-8f)) * grad->data[i];
    }
  }

  if (q_grad != NULL) {
    if (q_grad->rows != labels->rows || q_grad->cols != labels->cols) {
      return 0;
    }
    for (unsigned int i = 0; i < size; i++) {
      q_grad->data[i] += -logf(predictions->data[i] + 1e-8f) * grad->data[i];
    }
  }

  return 1;
}

void mat_free(matrix *m){
    free(m->data);
    free(m);
}
