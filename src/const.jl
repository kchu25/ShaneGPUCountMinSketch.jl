# default parameters
const int_type = Int32
const prime_number = int_type(50000101)
const default_cms_delta = 0.0001
const default_cms_epsilon = 0.00005
const default_min_count = 25

# CUDA setup
const default_num_threads1D = (128,)
const default_num_threads2D = (24,24)
const default_num_threads3D = (8,8,8)
