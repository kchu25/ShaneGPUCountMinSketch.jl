"""
input: 
    num_fils: the number of filters in the configuration
    max_nz_len: the maximum number of the non-zero code components
        (this needs to be counted first as to reduce the number of combinations)

"""
function generate_combinations_and_cms(
        num_fils::Integer, 
        max_nz_len::Integer;
        delta=default_cms_delta, epsilon=default_cms_epsilon)
    combs = int_type.(
        reduce(hcat, combinations(1:max_nz_len, num_fils) |> collect)) |> cu 
    c = gpu_cms(config_size(num_fils), delta, epsilon)
    # b/c (f₁, d₁₂, f₂, d₂₃, f₃); definition of configuration
    return combs, c
end

"""
    combs: the combinations of the non-zero code components
    c: the CountMinSketch object
    A: Array that contains the code
        A[:,1,n] contains the placements
        A[:,2,n] contains the corresponding filter indices
    fil_len: the length of the filter; supply by the user

modify the CountMinSketch object c in place

note: A could be a parition
"""
function count!(combs, c::gpu_cms, A::CuArray{int_type,3}, fil_len::Integer)
    size_tuple = (size(combs, 2), size(Sk, 1), size(A, 3))
    cms_M = size(c.Sk) |> prod
    cms_cols = size(c.Sk,2)
    # execute the counting on the sketch
    @cuda threads=default_num_threads3D blocks=ceil.(
            Int, size_tuple) count_kernel(
                combs, A, c.R, c.Sk, cms_M, cms_cols, fil_len);
end

"""
obtain placeholder_count
"""
function check!(configurations::Set{Vector{Int}}, 
        combs, combs_cpu, A, A_cpu, c::gpu_cms, num_fils, fil_len; 
        min_count=default_min_count)
    cms_M = size(c.Sk) |> prod
    cms_cols = size(c.Sk,2)
    size_tuple = (size(combs, 2), size(A, 3))
    placeholder_count = CUDA.fill(false, size_tuple); 
    # get the placeholder_count
    @cuda threads=default_num_threads2D blocks=ceil.(
        Int, size_tuple) count_kernel_chk(
            combs, A, c.R, c.Sk, cms_M, cms_cols, fil_len, placeholder_count,
            min_count)

    # now add the enriched configurations
    placeholder_count = placeholder_count |> BitArray
    configuration = Vector{Int}(undef, (config_size(num_fils),)) # TODO use static array
    for c in findall(placeholder_count .== true)
        i, n = c[1], c[2]
        comb_here = @view combs_cpu[:, i]
        # for loop to get the configuration
        for k in axes(comb_here, 1)
            configuration[2*(k-1)+1] = A_cpu[comb_here[k], 2, n]
            if k < size(comb_here, 1)
                configuration[2*k] = 
                    A_cpu[comb_here[k+1], 1, n] - A_cpu[comb_here[k], 1, n] - fil_len
            end
        end
        push!(configurations, copy(configuration))
    end
end
