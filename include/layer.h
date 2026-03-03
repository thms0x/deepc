#ifndef LAYER_H
#define LAYER_H
#include "model.h"

model_var *layer_fully_connected(model_context *model, model_var *input,
                                 unsigned int input_size,
                                 unsigned int output_size);

#endif
