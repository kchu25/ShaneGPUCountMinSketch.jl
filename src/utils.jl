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
    for i = 1:r.num_batches
        @cuda threads=default_num_threads3D blocks=ceil.(
                Int, get_sketch_size_tuple3d(r)) count_kernel(
                    r.combs_gpu, 
                    r.A_gpu[i], 
                    r.cms.R, 
                    r.cms.Sk, 
                    get_sketch_num_counters(r), 
                    get_sketch_num_cols(r), 
                    r.fil_len);
        @info "how many: $(sum(r.cms.Sk))"
    end
end

"""
obtain placeholder_count
"""
function check_and_fill_placeholder!(r::record;
        min_count=default_min_count)
    # get the placeholder_count
    for i = 1:r.num_batches
        @cuda threads=default_num_threads2D blocks=ceil.(
            Int, get_sketch_size_tuple2d(r)) count_kernel_chk(
                r.combs_gpu, 
                r.A_gpu[i], 
                r.cms.R, 
                r.cms.Sk, 
                get_sketch_num_counters(r), 
                get_sketch_num_cols(r), 
                r.fil_len, 
                r.placeholder_count[i],
                min_count)
    end
end

function _obtain_enriched_configurations_(r::record)
    #= get the configurations; (i.e. where 
        (combination, seq) in the placeholder 
        from the sketch exceed min_count) =#

    enriched_configs = Vector{Set{Tuple}}(undef, r.num_batches)
    for i = 1:r.num_batches
        @info "obtaining enriched configurations for batch $i"
        where_exceeds = findall(r.placeholder_count[i] .== true)
        configs = CuMatrix{int_type}(undef, (length(where_exceeds), config_size(r.num_fils)));

        @info "grid size: $(length(where_exceeds))"
        if length(where_exceeds) == 0
            enriched_configs[i] = Set{Vector{Int}}()
        else
            @cuda threads=default_num_threads1D blocks=ceil(Int, length(where_exceeds)) obtain_configs!(
                where_exceeds, r.combs_gpu, r.A_gpu[i], configs, r.fil_len)
            enriched_configs[i] = map(x->Tuple(x), eachrow(Array(configs))) |> Set
        end
    end

    return reduce(union, enriched_configs)
end

function obtain_enriched_configurations(
        nz_dict::Dict{Int, Vector{CartesianIndex{2}}},
        num_fils::Int, 
        fil_len::Int;
        min_count=default_min_count,
        delta = default_cms_delta,
        epsilon = default_cms_epsilon
        )
    r = record(nz_dict, num_fils, fil_len; delta=delta, epsilon=epsilon)
    count!(r)
    check_and_fill_placeholder!(r; min_count=min_count)
    configs =_obtain_enriched_configurations_(r)
    return configs
end