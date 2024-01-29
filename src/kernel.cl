__constant float X_FLOAT_REPRESENTATION = 11.0f;
__constant float ADD_FLOAT_REPRESENTATION = 1.0f;
__constant float SUB_FLOAT_REPRESENTATION = 2.0f;
__constant float MUL_FLOAT_REPRESENTATION = 3.0f;
__constant float DIV_FLOAT_REPRESENTATION = 4.0f;

__constant float NORMALIZATION_VAL = 255.0f;

// Input MUST BE aligned at a power of 2's size
__kernel void parallel_prefix_sum(__global float *input, int idx_offset) {
    int idx = (get_global_id(0) + 1) * idx_offset - 1;
    int idx_next = idx - idx_offset / 2;

    // printf("IDX=%d, IDX2=%d => %f with %f\n", idx, idx_next, input[idx], input[idx_next]);
    input[idx] = input[idx] + input[idx_next]; // Overall sum stored in the last element
}


__kernel void calculate_correlation_acc_values(__global float *nominator, __global float *acc_diffs_squared, __global float* hr_values_diffs, float avg_acc) {
  size_t id = get_global_id(0);

  float acc_diff = nominator[id] - avg_acc;
  nominator[id] = acc_diff * hr_values_diffs[id];
  acc_diffs_squared[id] = acc_diff * acc_diff;
}

__kernel void generate_hr_values(__global float* generation, int row_idx, int nodes_count, __global float* acc_values, __global float* generated_values) {
  const size_t id = get_global_id(0);
  const size_t node_size = 3; // parent, left and right child

  float val = 0.0f;

  for(int i = 0; i < nodes_count; i += node_size) {
    size_t idx = (row_idx * nodes_count * node_size) + i;
    float op = generation[idx]; 

    float op_l = generation[idx + 1]; 
    if (op_l == X_FLOAT_REPRESENTATION){
      op_l = acc_values[id];
    }

    float op_r = generation[idx + 2]; 
    
    // printf("[%d]: %f, [%d]: %f, [%d]: %f\n", idx, op, idx + 1, op_l, idx + 2, op_r);
    if(op == ADD_FLOAT_REPRESENTATION){ 
      val += (op_l + op_r);
    }
    else if(op == SUB_FLOAT_REPRESENTATION){
      val += (op_l - op_r);
    }
    else if(op == MUL_FLOAT_REPRESENTATION){
      val += (op_l * op_r);
    }
    else if(op == DIV_FLOAT_REPRESENTATION){
      if(op_r == 0.0f) { // Prevent zero division 
        val += op_l;
      }
      else{
        val += (op_l / op_r);
      }
    }
    else {
      // Should not happen, indicates that an invalid operation was generated
    }
  }
  
  generated_values[id] = val;
  // printf("GPU %d done, new value: %f\n", id, generated_values[id]);
}

__kernel void perform_crossover(__global float* generation, int crossover_idx, int nodes_count){
  const size_t id = get_global_id(0);
  const size_t node_size = 3;

  const size_t idx = id * 2 * nodes_count; // Skip over two

  float tmp = 0.0f;
  size_t first_idx = 0;
  size_t second_idx = 0;
  for(int i = crossover_idx; i < nodes_count; i += node_size){
    first_idx = idx + i;
    second_idx = idx + nodes_count + i;

    tmp = generation[first_idx];
    generation[first_idx] = generation[second_idx];
    generation[second_idx] = tmp;
   }
}
