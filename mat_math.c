#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

typedef enum {
  MV_FLAG_NONE = 0,
  MV_FLAG_REQUIRES_GRAD = (1 << 0),
  MV_FLAG_PARAMETER = (1 << 1),
  MV_FLAG_INPUT = (1 << 2),
  MV_FLAG_OUTPUT = (1 << 3),
  MV_FLAG_DESIRED_OUPUT = (1 << 4),
  MV_FLAG_COST = (1 << 5)

} model_var_flags;

typedef enum {
  MV_OP_NULL = 0,
  MV_OP_CREATE,

  _MV_OP_UNARY_START,
  MV_OP_RELU,
  MV_OP_SOFTMAX,

  _MV_OP_BINARY_START,
  MV_OP_ADD,
  MV_OP_SUB,
  MV_OP_MATMUL,
  MV_OP_CROSS_ENTROPY_LOSS,
} model_var_op;

#define MV_MODEL_VAR_MAX_INPUTS 2
#define MV_NUM_INPUTS(op)                                                      \
  ((op) < _MV_OP_UNARY_START ? 0 : (op) < _MV_OP_BINARY_START ? 1 : 2)

typedef struct model_var {
  int index;
  unsigned int flags;

  matrix *val;
  matrix *grad;

  model_var_op op;
  struct model_var *inputs[MV_MODEL_VAR_MAX_INPUTS];
} model_var;

typedef struct {
  model_var **vars;
  unsigned int size;
} model_program;

typedef struct {
  unsigned int num_vars;

  model_var *input;
  model_var *output;
  model_var *desired_output;
  model_var *cost;

  model_program forward_prog;
  model_program cost_prog;
} model_context;

typedef struct {
  matrix *train_images;
  matrix *train_labels;
  matrix *test_images;
  matrix *test_labels;

  unsigned int epochs;
  unsigned int batch_size;
  float learning_rate;
} model_training_desc;

model_var *mv_create(model_context *model, unsigned int rows, unsigned int cols,
                     unsigned int flags);
model_var *mv_relu(model_context *model, model_var *input, unsigned int flags);
model_var *mv_softmax(model_context *model, model_var *input,
                      unsigned int flags);
model_var *mv_add(model_context *model, model_var *a, model_var *b,
                  unsigned int flags);
model_var *mv_sub(model_context *model, model_var *a, model_var *b,
                  unsigned int flags);
model_var *mv_matmul(model_context *model, model_var *a, model_var *b,
                     unsigned int flags);
model_var *mv_cross_entropy(model_context *model, model_var *predictions,
                            model_var *labels, unsigned int flags);

model_program model_prog_create(model_context *model, model_var *out_var);
void model_prog_compute(model_program *prog);
void model_prog_compute_grad(model_program *prog);
model_context *model_create();
void model_compile(model_context *model);
void model_feedforward(model_context *model);
void model_train(model_context *model, const model_training_desc *desc);

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

int mat_cross_entropy_add_grad(const matrix *p_grad, const matrix *q_grad,
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

  printf("pre training output: \n");
  for (unsigned int i = 0; i < 10; i++) {
    printf("%f ", model->output->val->data[i]);
  }
  printf("\n");

  model_training_desc training_desc = {.train_images = train_images,
                                       .train_labels = train_labels,
                                       .test_images = test_images,
                                       .test_labels = test_labels,

                                       .epochs = 3,
                                       .batch_size = 20,
                                       .learning_rate = 0.005};
  model_train(model, &training_desc);

  memcpy(model->input->val->data, train_images->data, sizeof(float) * 784);

  model_feedforward(model);

  printf("post training output: \n");

  _draw_mnist_digit(train_images->data + 0 * 784);
  for (unsigned int i = 0; i < 10; i++) {
    printf("%f ", model->output->val->data[i]);
  }
  printf("\n");
  return 0;
}

model_var *mv_create(model_context *model, unsigned int rows, unsigned int cols,
                     unsigned int flags) {
  model_var *var = (model_var *)malloc(sizeof(model_var));

  var->index = model->num_vars++;
  var->flags = flags;
  var->val = mat_create(rows, cols);
  var->op = MV_OP_CREATE;

  if (flags & MV_FLAG_REQUIRES_GRAD) {
    var->grad = mat_create(rows, cols);
  }

  if (flags & MV_FLAG_OUTPUT) {
    model->output = var;
  }
  if (flags & MV_FLAG_INPUT) {
    model->input = var;
  }
  if (flags & MV_FLAG_DESIRED_OUPUT) {
    model->desired_output = var;
  }
  if (flags & MV_FLAG_COST) {
    model->cost = var;
  }
  return var;
}

model_var *_mv_unary_impl(model_context *model, model_var *input,
                          unsigned int flags, unsigned int rows,
                          unsigned int cols, model_var_op op) {
  if (op >= _MV_OP_BINARY_START || op <= _MV_OP_UNARY_START)
    fprintf(stderr, "Invalid operation passed to function\n");

  if (input->flags & MV_FLAG_REQUIRES_GRAD) {
    flags |= MV_FLAG_REQUIRES_GRAD;
  }
  model_var *out = mv_create(model, rows, cols, flags);
  out->op = op;
  out->inputs[0] = input;

  return out;
}

model_var *_mv_binary_impl(model_context *model, model_var *a, model_var *b,
                           unsigned int flags, unsigned int rows,
                           unsigned int cols, model_var_op op) {
  if (op < _MV_OP_BINARY_START)
    fprintf(stderr, "Invalid operation passed to function\n");

  if (a->flags & MV_FLAG_REQUIRES_GRAD || b->flags & MV_FLAG_REQUIRES_GRAD) {
    flags |= MV_FLAG_REQUIRES_GRAD;
  }
  model_var *out = mv_create(model, rows, cols, flags);
  out->op = op;
  out->inputs[0] = a;
  out->inputs[1] = b;

  return out;
}

model_var *mv_relu(model_context *model, model_var *input, unsigned int flags) {
  return _mv_unary_impl(model, input, flags, input->val->rows, input->val->cols,
                        MV_OP_RELU);
}
model_var *mv_softmax(model_context *model, model_var *input,
                      unsigned int flags) {
  return _mv_unary_impl(model, input, flags, input->val->rows, input->val->cols,
                        MV_OP_SOFTMAX);
}
model_var *mv_add(model_context *model, model_var *a, model_var *b,
                  unsigned int flags) {
  if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
    fprintf(stderr, "Input dimensions do not match for add operation\n");
    return NULL;
  }
  return _mv_binary_impl(model, a, b, flags, a->val->rows, a->val->cols,
                         MV_OP_ADD);
}

model_var *mv_sub(model_context *model, model_var *a, model_var *b,
                  unsigned int flags) {
  if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
    fprintf(stderr, "Input dimensions do not match for sub operation\n");
    return NULL;
  }
  return _mv_binary_impl(model, a, b, flags, a->val->rows, a->val->cols,
                         MV_OP_SUB);
}

model_var *mv_matmul(model_context *model, model_var *a, model_var *b,
                     unsigned int flags) {
  if (a->val->cols != b->val->rows) {
    fprintf(stderr, "Input dimensions do not match for matmul operation\n");
    return NULL;
  }
  return _mv_binary_impl(model, a, b, flags, a->val->rows, b->val->cols,
                         MV_OP_MATMUL);
}
model_var *mv_cross_entropy(model_context *model, model_var *predictions,
                            model_var *labels, unsigned int flags) {
  if (predictions->val->rows != labels->val->rows ||
      predictions->val->cols != labels->val->cols) {
    fprintf(stderr,
            "Input dimensions do not match for cross entropy operation\n");
    return NULL;
  }
  return _mv_binary_impl(model, predictions, labels, flags,
                         predictions->val->rows, predictions->val->cols,
                         MV_OP_CROSS_ENTROPY_LOSS);
}

model_program model_prog_create(model_context *model, model_var *out_var) {
  unsigned int *visited =
      (unsigned int *)calloc(model->num_vars, sizeof(unsigned int));
  unsigned int stack_size = 0;
  unsigned int out_size = 0;
  model_var **stack =
      (model_var **)malloc(model->num_vars * sizeof(model_var *));
  model_var **out = (model_var **)malloc(model->num_vars * sizeof(model_var *));
  // To be optimized
  stack[stack_size++] = out_var;
  while (stack_size > 0) {
    model_var *cur = stack[--stack_size];

    if (cur->index >= model->num_vars)
      continue;
    if (visited[cur->index]) {
      if (out_size < model->num_vars) {
        out[out_size++] = cur;
      }
      continue;
    }
    visited[cur->index] = 1;
    stack[stack_size++] = cur;
    unsigned int num_inputs = MV_NUM_INPUTS(cur->op);
    for (int i = 0; i < num_inputs; i++) {
      model_var *input = cur->inputs[i];

      if (input->index >= model->num_vars || visited[input->index])
        continue;

      for (unsigned int j = 0; j < stack_size; j++) {
        if (stack[j] == input) {
          for (unsigned int k = j; k < stack_size - 1; k++) {
            stack[k] = stack[k + 1];
          }
          stack_size--;
        }
      }
      stack[stack_size++] = input;
    }
  }

  model_program prog = {.size = out_size, .vars = out};
  return prog;
}

void model_prog_compute(model_program *prog) {
  for (unsigned int i = 0; i < prog->size; i++) {
    model_var *cur = prog->vars[i];
    model_var *a = cur->inputs[0];
    model_var *b = cur->inputs[1];

    switch (cur->op) {
    case MV_OP_CREATE:
    case MV_OP_NULL:
      break;
    case _MV_OP_UNARY_START:
      break;
    case MV_OP_RELU: {
      mat_relu(cur->val, a->val);
      break;
    };
    case MV_OP_SOFTMAX: {
      mat_softmax(cur->val, a->val);
      break;
    };
    case _MV_OP_BINARY_START:
      break;
    case MV_OP_ADD: {
      mat_add(cur->val, a->val, b->val);
      break;
    }
    case MV_OP_SUB: {
      mat_sub(cur->val, a->val, b->val);
      break;
    }
    case MV_OP_MATMUL: {
      mat_mul(cur->val, a->val, b->val, 1, 0, 0);
      break;
    }
    case MV_OP_CROSS_ENTROPY_LOSS: {
      mat_cross_entropy(cur->val, a->val, b->val);
      break;
    }
    }
  }
}

void model_prog_compute_grad(model_program *prog) {
  for (int i = 0; i < prog->size; i++) {
    model_var *cur = prog->vars[i];
    if ((cur->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD) {
      continue;
    }
    if ((cur->flags & MV_FLAG_PARAMETER)) {
      continue;
    }
    mat_clear(cur->grad);
  }
  mat_fill(prog->vars[prog->size - 1]->grad, 1.0f);

  for (int i = (int)prog->size - 1; i >= 0; i--) {
    model_var *cur = prog->vars[i];
    model_var *a = cur->inputs[0];
    model_var *b = cur->inputs[1];

    unsigned int num_inputs = MV_NUM_INPUTS(cur->op);
    if (num_inputs == 1 &&
        (a->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD)
      continue;
    if (num_inputs == 2 &&
        (a->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD &&
        (a->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD)
      continue;

    switch (cur->op) {
    case MV_OP_CREATE:
    case MV_OP_NULL:
      break;
    case _MV_OP_UNARY_START:
      break;
    case MV_OP_RELU: {
      mat_relu_add_grad(a->grad, a->val, cur->grad);
      break;
    };
    case MV_OP_SOFTMAX: {
      mat_softmax_add_grad(a->grad, cur->val, cur->grad);
      break;
    };
    case _MV_OP_BINARY_START:
      break;
    case MV_OP_ADD: {
      if (a->flags & MV_FLAG_REQUIRES_GRAD) {
        mat_add(a->grad, a->grad, cur->grad);
      }
      if (b->flags & MV_FLAG_REQUIRES_GRAD) {
        mat_add(b->grad, b->grad, cur->grad);
      }
      break;
    }
    case MV_OP_SUB: {
      if (a->flags & MV_FLAG_REQUIRES_GRAD) {
        mat_sub(a->grad, a->grad, cur->grad);
      }
      if (b->flags & MV_FLAG_REQUIRES_GRAD) {
        mat_sub(b->grad, b->grad, cur->grad);
      }
      break;
    }
    case MV_OP_MATMUL: {
      if (a->flags & MV_FLAG_REQUIRES_GRAD) {
        mat_mul(a->grad, cur->grad, b->val, 0, 0, 1);
      }
      if (b->flags & MV_FLAG_REQUIRES_GRAD) {
        mat_mul(b->grad, a->val, cur->grad, 0, 1, 0);
      }
      break;
    }
    case MV_OP_CROSS_ENTROPY_LOSS: {
      model_var *p = a;
      model_var *q = b;
      mat_cross_entropy_add_grad(p->grad, q->grad, p->val, q->val, cur->grad);
      break;
    }
    }
  }
}
model_context *model_create() {
  return (model_context *)malloc(sizeof(model_context));
}
void model_compile(model_context *model) {
  if (model->output != NULL) {
    model->forward_prog = model_prog_create(model, model->output);
  }

  if (model->cost != NULL) {
    model->cost_prog = model_prog_create(model, model->cost);
  }
}
void model_feedforward(model_context *model) {
  model_prog_compute(&model->forward_prog);
}
void model_train(model_context *model,
                 const model_training_desc *training_desc) {
  // stochastic grad descent

  matrix *train_images = training_desc->train_images;
  matrix *train_labels = training_desc->train_labels;
  matrix *test_images = training_desc->test_images;
  matrix *test_labels = training_desc->test_labels;

  unsigned int num_examples = train_images->rows;
  unsigned int input_size = train_images->cols;
  unsigned int num_tests = test_images->rows;
  unsigned int output_size = train_labels->cols;

  unsigned int num_batches = num_examples / training_desc->batch_size;

  unsigned int *training_order =
      (unsigned int *)calloc(sizeof(unsigned int), num_examples);
  for (unsigned int i = 0; i < num_examples; i++) {
    training_order[i] = i;
  }

  for (unsigned int epoch = 0; epoch < training_desc->epochs; epoch++) {
    for (unsigned int i = 0; i < num_examples; i++) {
      unsigned int a = rand() % num_examples;
      unsigned int b = rand() % num_examples;

      unsigned int tmp = training_order[b];
      training_order[b] = training_order[a];
      training_order[a] = tmp;
    }

    for (unsigned int batch = 0; batch < num_batches; batch++) {
      for (unsigned int i = 0; i < model->cost_prog.size; i++) {
        model_var *cur = model->cost_prog.vars[i];

        if (cur->flags & MV_FLAG_PARAMETER) {
          mat_clear(cur->grad);
        }
      }
      float avg_cost = 0.0f;
      for (unsigned int i = 0; i < training_desc->batch_size; i++) {
        unsigned int order_index = batch * training_desc->batch_size + i;
        unsigned int index = training_order[order_index];

        memcpy(model->input->val->data, train_images->data + index * input_size,
               sizeof(float) * input_size);

        memcpy(model->desired_output->val->data,
               train_labels->data + index * output_size,
               sizeof(float) * output_size);

        model_prog_compute(&model->cost_prog);
        model_prog_compute_grad(&model->cost_prog);

        avg_cost += mat_sum(model->cost->val);
      }
      avg_cost /= (float)training_desc->batch_size;

      for (unsigned int i = 0; i < model->cost_prog.size; i++) {
        model_var *cur = model->cost_prog.vars[i];

        if ((cur->flags & MV_FLAG_PARAMETER) != MV_FLAG_PARAMETER) {
          continue;
        }

        mat_scale(cur->grad,
                  training_desc->learning_rate / training_desc->batch_size);
        mat_sub(cur->val, cur->val, cur->grad);
      }
      printf("Epoch %2d /%2d, Batch %4d / %4d, Average Cost: %.4f\r", epoch + 1,
             training_desc->epochs, batch + 1, num_batches, avg_cost);
    }
    printf("\n");

    unsigned int num_correct = 0;
    float avg_cost = 0;
    for (unsigned int i = 0; i < num_tests; i++) {
      memcpy(model->input->val->data, test_images->data + i * input_size,
             sizeof(float) * input_size);

      memcpy(model->desired_output->val->data,
             test_labels->data + i * output_size, sizeof(float) * output_size);

      model_prog_compute(&model->cost_prog);
      avg_cost += mat_sum(model->cost->val);
      num_correct += mat_argmax(model->output->val) ==
                     mat_argmax(model->desired_output->val);
    }
    avg_cost /= (float)num_tests;
    printf("Test Completed. Accuracy: %5d / %d (%.1f%%), Average Cost %.4f\n",
           num_correct, num_tests, (float)num_correct / num_tests * 100.0f,
           avg_cost);
  }
}
