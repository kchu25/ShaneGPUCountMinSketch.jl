
function count_kernel(combs, A, R, Sk, M, ncols, pfm_len)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y;
    n = (blockIdx().z - 1) * blockDim().z + threadIdx().z;

    I = size(combs, 2)
    J = size(Sk, 1)
    N = size(A, 3)
    if i ≤ I && j ≤ J && n ≤ N

        @inbounds for k in axes(combs, 1)
            if A[combs[k,i], 1, n] == 0
                return nothing
            end
        end        
        num_here = 0
        for k in axes(combs, 1)
            # get the filter number and times the random number
            num_here += A[combs[k,i], 2, n] * R[j, 2*(k-1)+1] # filter
            if k < size(combs,1)
                # calculate the distance between two filters
                _distance_ =  A[combs[k+1,i],1,n] - A[combs[k,i],1,n] - pfm_len
                # don't take overlapping filters into account
                if _distance_ < 0
                    return nothing
                end
                # times the random number
                num_here += R[j, 2*k] * _distance_
            end
        end
        # get the column index; +1 to adjust to 1-base indexing
        num_here = ((num_here % M) % ncols) + 1
        # counter increment
        CUDA.@atomic Sk[j, num_here] += 1
    end
    return nothing
end

function count_kernel_chk(combs, A, R, Sk, M, ncols, pfm_len, placeholder_count, default_max_freq_member)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    n = (blockIdx().y - 1) * blockDim().y + threadIdx().y;

    I = size(combs, 2)
    J = size(Sk, 1)
    N = size(A, 3)
    if i ≤ I && n ≤ N
        @inbounds for k in axes(combs, 1)
            if A[combs[k,i], 1, n] == 0
                return nothing
            end
        end
        # placeholder_count[i, n] = true
        exceed = true
        for j = 1:J
            num_here = 0
            for k in axes(combs, 1)
                # get the filter number and times the random number
                num_here += A[combs[k,i], 2, n] * R[j, 2*(k-1)+1] # filter
                if k < size(combs,1)
                    # calculate the distance between two filters
                    _distance_ =  A[combs[k+1,i],1,n] - A[combs[k,i],1,n] - pfm_len
                    if _distance_ < 0
                        return nothing
                    end
                    # times the random number
                    num_here += R[j, 2*k] * _distance_
                end
            end
            # get the column index; +1 to adjust to 1-base indexing
            num_here = (num_here % M) % ncols + 1
            if  1 ≤ num_here ≤ size(Sk, 2)
                if Sk[j, num_here] < default_max_freq_member
                    exceed = false
                end
            end
        end
        placeholder_count[i, n] = exceed
    end
    return nothing
end

