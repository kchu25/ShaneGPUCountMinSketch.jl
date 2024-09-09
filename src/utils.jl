
"""
    combs_gpu: the combinations of the non-zero code components
    c: the CountMinSketch object
    A: Array that contains the code
        A[:,1,n] contains the placements
        A[:,2,n] contains the corresponding filter indices
    fil_len: the length of the filter; supply by the user

modify the CountMinSketch object c in place

note: A could be a parition (TODO)
"""
function count!(r::record)
    # execute the counting on the sketch
    @cuda threads=default_num_threads3D blocks=ceil.(
            Int, get_sketch_size_tuple3d(r)) count_kernel(
                r.combs_gpu, 
                r.A_gpu, 
                r.cms.R, 
                r.cms.Sk, 
                get_sketch_num_counters(r), 
                get_sketch_num_cols(r), 
                r.fil_len);
end

"""
obtain placeholder_count
"""
function check_and_fill_placeholder!(r::record;
        min_count=default_min_count)
    # get the placeholder_count
    @cuda threads=default_num_threads2D blocks=ceil.(
        Int, get_sketch_size_tuple2d(r)) count_kernel_chk(
            r.combs_gpu, 
            r.A_gpu, 
            r.cms.R, 
            r.cms.Sk, 
            get_sketch_num_counters(r), 
            get_sketch_num_cols(r), 
            r.fil_len, 
            r.placeholder_count,
            min_count)
end

function _obtain_enriched_configurations_(r::record)
    #= get the configurations; (i.e. where 
        (combination, seq) in the placeholder 
        from the sketch exceed min_count) =#
    @time where_exceeds = findall(r.placeholder_count)
    # note: findall is slow and compiles each time
    # has to be done "in the loop" to avoid re-compilation TODO fix this
    if isempty(where_exceeds)
        return Set{Vector{Int}}()
    end

    configs = CuMatrix{int_type}(undef, (length(where_exceeds), config_size(r.num_fils)));
    @cuda threads=default_num_threads1D blocks=ceil(Int, length(where_exceeds)) obtain_configs!(
        where_exceeds, r.combs_gpu, r.A_gpu, configs, r.fil_len)

    return map(x->Tuple(x), eachrow(Array(configs))) |> Set
end

# function obtain_enriched_configurations2(r::record)
#     # make the set
#     configurations = Set{Vector{Int}}()
#     configuration = Vector{Int}(undef, (config_size(r.num_fils),)) # TODO use static array

#     # now add the enriched configurations
#     # i.e. all the configurations that have the counts larger than min_count (see const.jl)
#     placeholder_count_bitarr = r.placeholder_count |> Array
#     for c in findall(placeholder_count_bitarr .== true)
#         i, n = c[1], c[2]
#         comb_here = @view r.combs_cpu[:, i]
#         # for loop to get the configuration
#         for k in axes(comb_here, 1)
#             configuration[2*(k-1)+1] = r.A_cpu[comb_here[k], 2, n]
#             if k < size(comb_here, 1)
#                 configuration[2*k] = 
#                     r.A_cpu[comb_here[k+1], 1, n] - r.A_cpu[comb_here[k], 1, n] - r.fil_len
#             end
#         end
#         push!(configurations, copy(configuration))
#     end
#     return configurations
# end

function obtain_enriched_configurations(
        nz_dict::Dict{Int, Vector{CartesianIndex{2}}},
        num_fils::Int, 
        fil_len::Int;
        min_count=default_min_count
        )
    r = record(nz_dict, num_fils, fil_len)
    count!(r)
    check_and_fill_placeholder!(r; min_count=min_count)
    configs =_obtain_enriched_configurations_(r)
    return configs
end