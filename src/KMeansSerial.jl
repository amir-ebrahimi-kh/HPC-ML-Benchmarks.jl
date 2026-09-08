module KMeansSerial

export kmeans_serial

"""
    kmeans_serial(X::Matrix{Float64}, k::Int, max_iters::Int=100)

Naive serial implementation of K-Means clustering.
`X` is a matrix of shape (D, N) where D is the number of features and N is the number of points.
`k` is the number of clusters.
"""
function kmeans_serial(X::Matrix{Float64}, k::Int, max_iters::Int=100)
    D, N = size(X)

    # Initialize centroids by randomly selecting k points from X
    centroids = X[:, rand(1:N, k)]

    # Pre-allocate array for assignments
    assignments = zeros(Int, N)

    # Loop over maximum iterations
    for iter in 1:max_iters
        changed = false

        # Step 1: Assignment
        for i in 1:N
            min_dist = Inf
            best_cluster = 1

            for j in 1:k
                # Calculate squared Euclidean distance
                dist = 0.0
                for d in 1:D
                    diff = X[d, i] - centroids[d, j]
                    dist += diff * diff
                end

                if dist < min_dist
                    min_dist = dist
                    best_cluster = j
                end
            end

            if assignments[i] != best_cluster
                assignments[i] = best_cluster
                changed = true
            end
        end

        # Check for convergence
        if !changed
            break
        end

        # Step 2: Update centroids
        new_centroids = zeros(Float64, D, k)
        counts = zeros(Int, k)

        for i in 1:N
            cluster = assignments[i]
            counts[cluster] += 1
            for d in 1:D
                new_centroids[d, cluster] += X[d, i]
            end
        end

        for j in 1:k
            if counts[j] > 0
                for d in 1:D
                    centroids[d, j] = new_centroids[d, j] / counts[j]
                end
            end
        end
    end

    return centroids, assignments
end

end # module