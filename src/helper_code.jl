
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
        reduce(hcat, combinations(1:max_nz_len, num_fils) |> collect)) 
    return combs, combs |> cu
end

function make_gpu_cms(num_fils; 
        delta=default_cms_delta, epsilon=default_cms_epsilon)
    # config_size: return the dimension of the configuration 
    # according to the number of filters
    c = gpu_cms(config_size(num_fils), delta, epsilon)
    return c
end

"""
get_A!(nz_dict::Dict{Int, Vector{CartesianIndex{2}}})
    Take non-zero code components
        i.e. Dict{Int => Vector{CartesianIndex{2}}}
                  seq => [(pos, fil)]
    and convert it to an Array of records

In another word: construct an Array A: Array that contains the code
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
    num_seqs = keys(nz_dict) |> length 

    num_batches_m1 = num_seqs ÷ batch_size # number of batches minus 1
    last_batch_size = num_seqs % batch_size
    num_batches = num_batches_m1 + 1
    ##### construct A #####
    A = [zeros(int_type, (max_nz_len,2,batch_size)) for _ in 1:num_batches_m1]
    push!(A, zeros(int_type, (max_nz_len,2,last_batch_size)))

    for (ind1, seq) in enumerate(keys(nz_dict))
        batch_ind = ind1 ÷ batch_size + 1
        placement_ind = ind1 % batch_size
        for (ind2, c) in enumerate(nz_dict[seq])
            pos, fil = c[1], c[2]
            A[batch_ind][ind2, 1, placement_ind] = pos
            A[batch_ind][ind2, 2, placement_ind] = fil
        end
    end
    @info "A1: $(A[1][:,:,1])"
    return A, cu.(A), num_batches
end
