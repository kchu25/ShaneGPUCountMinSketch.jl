module ShaneGPUCountMinSketch

using CUDA
using Combinatorics

include("const.jl")
include("helper.jl")
include("sketch.jl")
include("kernel.jl")
include("helper_code.jl")
include("record_struct.jl")
include("utils.jl")

export obtain_enriched_configurations

end
