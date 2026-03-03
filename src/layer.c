#include "model.h"
#include "math.h"

model_var* layer_fully_connected(model_context *model, model_var *input,
                           unsigned int input_size, unsigned int output_size
                           ) {
  model_var *weights =
      mv_create(model, output_size, input_size, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  model_var *biases = mv_create(model, output_size, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

  float bound = sqrtf(6.0f / (input_size + output_size));

  mat_fill_rand(weights->val, -bound, bound);
  mat_fill(biases->val, 0);

  model_var *matmul = mv_matmul(model, weights, input, 0);
  model_var *add = mv_add(model, matmul, biases, 0);
  return add;
}
