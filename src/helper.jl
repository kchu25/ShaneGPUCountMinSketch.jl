# number of rows in the sketch
cms_rows(cms_delta) = ceil(log(1 / cms_delta)) |> Int
# number of counters in the sketch
cms_num_counters(rows, cms_epsilon) = 
    rows * ceil(exp(1) / cms_epsilon) |> Int
# number of columns in the sketch
cms_cols(num_counters, rows) = num_counters ÷ rows

# get the number of rows, counters, and columns for the sketch
function get_num_rows_counters_cols(delta, epsilon)
    rows = cms_rows(delta)
    num_counters = cms_num_counters(rows, epsilon)
    cols = cms_cols(num_counters, rows)
    @info "Sketch information:"
    @info "number of rows: $rows"
    @info "number of columns: $cols"
    return rows, num_counters, cols
end

config_size(num_fils::Integer) = 2*num_fils - 1
