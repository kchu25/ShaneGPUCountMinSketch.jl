"""
d: the dimension of the configuration
    e.g. for a configuration of 3 elements, d = 3
    e.g. (f₁, d₁₂, f₂) config of length 3
         (f₁, d₁₂, f₂, d₂₃, f₃) config of length 5
"""
mutable struct gpu_cms
    R::CuMatrix{int_type}
    Sk::CuMatrix{int_type}
    function gpu_cms(d::Integer)
        rows, num_counters, cols = 
            get_num_rows_counters_cols(
                default_cms_delta, default_cms_epsilon)
        R = int_type.(rand(1:num_counters-1, (rows, d))) |> cu
        Sk = zeros(int_type, (rows, cols)) |> cu
        new(R, Sk)
    end
    function gpu_cms(d::Integer, delta::Float64, epsilon::Float64)
        rows, num_counters, cols = 
            get_num_rows_counters_cols(delta, epsilon)
        R = int_type.(rand(1:num_counters-1, (rows, d))) |> cu
        Sk = zeros(int_type, (rows, cols)) |> cu
        new(R, Sk)
    end
end