
"""
filter_empty_seq!(nz_dict::Dict{Int, Vector{CartesianIndex{2}}})
    Filter out keys (seq) that have empty values
"""
function filter_empty_seq!(nz_dict::Dict{Int, Vector{CartesianIndex{2}}})
    for k in keys(nz_dict)
        isempty(nz_dict[k]) && delete!(nz_dict, k)
    end
end

"""
Get maximum number of non-zero code components in our records
    This is to construct the upperbound of the Array of records
"""
function get_max_nz_len(nz_dict::Dict{Int, Vector{CartesianIndex{2}}})        
    max_nz_len = 0 
    for k in keys(nz_dict)
        max_nz_len = max(max_nz_len, 
            length(nz_dict[k]))
    end
    return max_nz_len
end

"""
input: 
    num_fils: the number of filters in the configuration
    max_nz_len: the maximum number of the non-zero code components
        (this needs to be counted first as to reduce the number of combinations)

"""
function generate_combinations(
        num_fils::Integer, 
        max_nz_len::Integer;)
    combs = int_type.(
        reduce(hcat, combinations(1:max_nz_len, num_fils) |> collect)) |> cu 

    return combs, c
end

function make_gpu_cms(num_fils; 
        delta=default_cms_delta, epsilon=default_cms_epsilon)
    c = gpu_cms(config_size(num_fils), delta, epsilon)
    return c
end

"""
get_A!(nz_dict::Dict{Int, Vector{CartesianIndex{2}}})
    Take non-zero code components
        i.e. Dict{Int => Vector{CartesianIndex{2}}}
                  seq => [(pos, fil)]
    and convert it to an Array of records

In another word: construct the Array A: Array that contains the code
    A[:,1,n] contains the placements
    A[:,2,n] contains the corresponding filter indices

In the process it also deletes the empty seqs in nz_dict

Note: can partition nz_dict
    into the 1st 1000 entries, 2nd 1000 entries, etc...
        in case the memory usage is a concern

returns the cpu and gpu version of A
    i.e. returns A_cpu, A_gpu
"""
function get_A_and_combs!(nz_dict::Dict{Int, Vector{CartesianIndex{2}}}, max_nz_len)
    ##### preprocessing #####
    # filter out seq with empty values
    filter_empty_seq!(nz_dict)
    # get the number of seqs
    klen = keys(code_profile.nz_components_train) |> length 

    ##### construct A #####
    A = zeros(int_type, (max_nz_len,2,klen))
    for (ind1, seq) in enumerate(keys(nz_dict))
        for (ind2, c) in enumerate(nz_dict[seq])
            pos, fil = c[1], c[2]
            A[ind2, 1, ind1] = pos
            A[ind2, 2, ind1] = fil
        end
    end
    return A, cu(A)
end


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

TODO: Make A and placeholder a vector later on
    in case the memory usage is a concern
"""

mutable struct record
    A_cpu::Array{int_type, 3}
    A_gpu::CuArray{int_type, 3}
    combs::CuArray{int_type, 2}
    cms::gpu_cms # count min sketch
    placeholder_count::CuArray{Bool, 2}
    function record(nz_dict::Dict{Int, Vector{CartesianIndex{2}}}, 
                    num_fils; # the number of filters in the configuration
                    delta=default_cms_delta, epsilon=default_cms_epsilon)
        # maximum number of non-zero code components in each seq
        max_nz_len = get_max_nz_len(nz_dict)
        A_cpu, A_gpu = get_A_and_combs!(nz_dict, max_nz_len)
        combs = generate_combinations_and_cms(num_fils, max_nz_len)
        cms = make_gpu_cms(num_fils; delta=delta, epsilon=epsilon)
        placeholder_count = CUDA.fill(false, (size(combs, 2), size(A_cpu, 3))) |> cu
        new(A_cpu, A_gpu, combs, cms, placeholder_count)
    end
end
