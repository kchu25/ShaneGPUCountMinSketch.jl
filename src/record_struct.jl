"""
Take non-zero code components
    i.e. Dict{Int => Vector{CartesianIndex{2}}}
              seq => [(pos, fil)]
    and convert it to an Array of records

    nz_dict: non-zero code components
        Dict{Int => Vector{CartesianIndex{2}}}
        seq => [(pos, fil)]
    num_fils: the number of filters in the configuration
    # e.g. (f₁, d₁₂, f₂, d₂₃, f₃) has 3 filters

"""
mutable struct record
    A_cpu::Vector{Array{int_type, 3}}
    A_gpu::Vector{CuArray{int_type, 3}}
    num_batches::Int
    combs_cpu::Array{int_type, 2}
    combs_gpu::CuArray{int_type, 2}
    cms::gpu_cms # count min sketch
    placeholder_count::Vector{CuArray{Bool, 2}}
    num_fils::Int # the number of filters in the configuration
    fil_len::Int # the length of the filter in the model
    function record(nz_dict::Dict{Int, Vector{CartesianIndex{2}}}, 
                    num_fils, # the number of filters in the configuration
                    fil_len::Int; 
                    delta=default_cms_delta, 
                    epsilon=default_cms_epsilon,
                    batch_size=batch_size)
        # maximum number of non-zero code components in each seq
        max_nz_len = get_max_nz_len(nz_dict)
        A_cpu, A_gpu, num_batches = get_A_and_combs!(nz_dict, max_nz_len; batch_size=batch_size)
        combs, combs_gpu = generate_combinations(num_fils, max_nz_len)
        cms = make_gpu_cms(num_fils; delta=delta, epsilon=epsilon)
        placeholder_count = 
            [CUDA.fill(false, (size(combs, 2), size(A_cpu[i], 3))) for i = 1:num_batches]
        new(A_cpu, A_gpu, num_batches, combs, combs_gpu, cms, placeholder_count, num_fils, fil_len)
    end
end

get_sketch_num_counters(r::record) = size(r.cms.Sk) |> prod
get_sketch_num_cols(r::record) = size(r.cms.Sk, 2)
get_sketch_size_tuple3d(r::record, i::Int) = 
    (size(r.combs_cpu, 2), size(r.cms.Sk, 1), size(r.A_cpu[i], 3))
get_sketch_size_tuple2d(r::record, i::Integer) = (size(r.combs_cpu, 2), size(r.A_cpu[i], 3))
