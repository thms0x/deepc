#ifndef MODEL_H
#define MODEL_H

#include "matrix.h"

typedef enum model_var_flags{
  MV_FLAG_NONE = 0,
  MV_FLAG_REQUIRES_GRAD = (1 << 0),
  MV_FLAG_PARAMETER = (1 << 1),
  MV_FLAG_INPUT = (1 << 2),
  MV_FLAG_OUTPUT = (1 << 3),
  MV_FLAG_DESIRED_OUPUT = (1 << 4),
  MV_FLAG_COST = (1 << 5)

} model_var_flags;

typedef enum model_var_op{
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

typedef struct model_program{
  model_var **vars;
  unsigned int size;
} model_program;

typedef struct model_context{
  unsigned int num_vars;

  model_var *input;
  model_var *output;
  model_var *desired_output;
  model_var *cost;

  model_program forward_prog;
  model_program cost_prog;
} model_context;

typedef struct model_training_desc{
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

#endif