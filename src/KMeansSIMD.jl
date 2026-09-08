module KMeansSIMD

export kmeans_simd

"""
    kmeans_simd(X::Matrix{Float64}, k::Int, max_iters::Int=100)

SIMD optimized implementation of K-Means clustering.
`X` is a matrix of shape (D, N) where D is the number of features and N is the number of points.
`k` is the number of clusters.
"""
function kmeans_simd(X::Matrix{Float64}, k::Int, max_iters::Int=100)
    D, N = size(X)

    # Initialize centroids by randomly selecting k points from X
    centroids = X[:, rand(1:N, k)]

    # Pre-allocate array for assignments
    assignments = zeros(Int, N)

    # Pre-allocate space for centroids update to avoid allocation in inner loop
    new_centroids = zeros(Float64, D, k)
    counts = zeros(Int, k)

    for iter in 1:max_iters
        changed = false

        # Step 1: Assignment
        for i in 1:N
            min_dist = Inf
            best_cluster = 1

            for j in 1:k
                dist = 0.0
                @inbounds @simd for d in 1:D
                    diff = X[d, i] - centroids[d, j]
                    dist += diff * diff
                end

                if dist < min_dist
                    min_dist = dist
                    best_cluster = j
                end
            end

            @inbounds if assignments[i] != best_cluster
                assignments[i] = best_cluster
                changed = true
            end
        end

        if !changed
            break
        end

        # Step 2: Update centroids
        fill!(new_centroids, 0.0)
        fill!(counts, 0)

        @inbounds for i in 1:N
            cluster = assignments[i]
            counts[cluster] += 1
            @simd for d in 1:D
                new_centroids[d, cluster] += X[d, i]
            end
        end

        @inbounds for j in 1:k
            if counts[j] > 0
                @simd for d in 1:D
                    centroids[d, j] = new_centroids[d, j] / counts[j]
                end
            end
        end
    end

    return centroids, assignments
end

end # module