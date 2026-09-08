using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BenchmarkTools
using Random
using Printf

# Include the source files directly for this script
include("../src/KMeansSerial.jl")
include("../src/KMeansSIMD.jl")

using .KMeansSerial
using .KMeansSIMD

function run_benchmarks()
    # Number of features, number of clusters
    D = 8
    k = 5
    max_iters = 100

    println("==================================================")
    println("      K-Means Performance Benchmarks")
    println("==================================================\n")

    for N in [10_000, 100_000]
        println("Generating synthetic dataset with $N points ($D dimensions)...")
        Random.seed!(42)
        X = rand(D, N)

        println("\nBenchmarking Naive Serial Implementation ($N points)...")
        # Run once to force compilation
        kmeans_serial(X, k, max_iters)
        bm_serial = @benchmark kmeans_serial($X, $k, $max_iters) samples=5 seconds=10

        println("\nBenchmarking SIMD Optimized Implementation ($N points)...")
        # Run once to force compilation
        kmeans_simd(X, k, max_iters)
        bm_simd = @benchmark kmeans_simd($X, $k, $max_iters) samples=5 seconds=10

        display(bm_serial)
        println()
        display(bm_simd)
        println()

        t_serial = median(bm_serial).time
        t_simd = median(bm_simd).time
        speedup = t_serial / t_simd

        @printf("\n--> Speedup with SIMD (@inbounds & @simd) for %d points: %.2fx\n", N, speedup)
        println("--------------------------------------------------\n")
    end
end

run_benchmarks()