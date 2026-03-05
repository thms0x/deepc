#ifndef MATRIX_H
#define MATRIX_H

typedef struct matrix {
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
float mat_sum(matrix *m);
int mat_argmax(matrix *mat);
int mat_add(matrix *out, const matrix *a, const matrix *b);
int mat_sub(matrix *out, const matrix *a, const matrix *b);
int mat_mul(matrix *out, const matrix *a, const matrix *b, int zero_out,
            int transpose_a, int transpose_b);
void mat_fill(matrix *m, float x);
void mat_fill_rand(matrix *m, float lower, float upper);
int mat_relu(matrix *out, const matrix *in);
int mat_softmax(matrix *out, const matrix *in);
int mat_cross_entropy(matrix *out, const matrix *predictions,
                      const matrix *labels);
int mat_relu_add_grad(matrix *out, const matrix *in, const matrix *grad);
int mat_softmax_add_grad(matrix *out, const matrix *in, const matrix *grad);
int mat_cross_entropy_add_grad(matrix *p_grad, matrix *q_grad, const matrix *predictions,
                               const matrix *labels, const matrix *grad);
void mat_free(matrix *m);

#endif
