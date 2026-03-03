#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

model_var* mv_relu(model_context *model, model_var *input, unsigned int flags) {
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
