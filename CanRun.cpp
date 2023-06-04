#include <iostream>
#include <vector>
#include <random>
#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"
#include<cmath>
using namespace std;
using namespace oneapi;

constexpr int N = 3;
constexpr int iterations = 1000;
std::vector<double> solveLinearEquation(sycl::queue& q, const std::vector<std::vector<double>>& A, const std::vector<double>& B) {
    int64_t n = A.size();

    // Convert 2D vector to 1D vector
    std::vector<double> A_1D(n * n);
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < n; ++j)
            A_1D[i * n + j] = A[j][i];

    // Copy vectors to SYCL buffers
   auto A_buf = sycl::buffer<double, 1>(A_1D.data(), sycl::range<1>(n * n));
auto B_buf = sycl::buffer<double, 1>(B.data(), sycl::range<1>(n));
auto ipiv_buf = sycl::buffer<int64_t, 1>(sycl::range<1>(n)); 

    
    // Create scratchpad buffer
    auto scratchpad_size = oneapi::mkl::lapack::getrf_scratchpad_size<double>(q, n, n, n);
    sycl::buffer<double, 1> scratchpad_buf(scratchpad_size);

    // Call oneMKL LAPACK getrf function to compute the LU factorization of A
    oneapi::mkl::lapack::getrf(q, n, n, A_buf, n, ipiv_buf, scratchpad_buf, scratchpad_size);
    oneapi::mkl::lapack::getrs(q, oneapi::mkl::transpose::nontrans, n, 1, A_buf, n, ipiv_buf, B_buf, n, scratchpad_buf, scratchpad_size);

    // Copy result from SYCL buffer back to the std::vector
    std::vector<double> X(n);
    {
        auto B_acc = B_buf.get_access<sycl::access::mode::read>();
        for(int i = 0; i < n; ++i)
            X[i] = B_acc[i];
    }

    return X;
}
vector<vector<double>> generateMatrix(const vector<double>& constants, sycl::queue& q) {
    int n = constants.size()-1;
    vector<vector<double>> matrix(n, vector<double>(n, 0));

    sycl::buffer<double, 1> constants_buffer(constants.data(), sycl::range<1>(n));
    
    sycl::buffer<double, 2> matrix_buffer(sycl::range<2>(n, n));

    q.submit([&](sycl::handler& h) {
        auto constants_accessor = constants_buffer.get_access<sycl::access::mode::read>(h);
        auto matrix_accessor = matrix_buffer.get_access<sycl::access::mode::write>(h);
       
        h.parallel_for<class generateMatrix>(sycl::range<2>(n, n), [=](sycl::id<2> id) {
            int i = id[0];
            int j = id[1];

            if(j == i )
                matrix_accessor[id] = constants_accessor[i] + constants_accessor[i + 1];
            else if(j == i - 1 || j == i + 1)
                matrix_accessor[id] = -constants_accessor[(j>i)?j:i];
            else
                matrix_accessor[id] = 0;
        });
    });
    
    // Wait for all tasks to finish.
    q.wait();

    // Create a host_accessor after the queue has finished processing to ensure safe access
    sycl::host_accessor<double, 2> matrix_accessor(matrix_buffer);

    // Copy the result back to the host
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j){
      //   if(matrix[i][j])   cout<<" maxtrix is " << matrix[i][j] <<" " ;
            matrix[i][j] = matrix_accessor[i][j];
        }
    return matrix;
}
template<typename T>
void printMatrix(const vector<vector<T>>& matrix) {
    for (const auto& row : matrix) {
        for (T num : row) {
            cout << num << "\t";
        }
        cout << endl;
    }
}
tuple<double, double, double, double, double, vector<double>> calculate_statistics(sycl::queue& q, const vector<double>& X, int num_bins, double bin_width) {
    double sum = 0.0, square_sum = 0.0, cube_sum = 0.0;
    double min_val = numeric_limits<double>::max();
    double max_val = numeric_limits<double>::lowest();

    vector<int> bins(num_bins, 0);

    {
        sycl::buffer<double, 1> X_buf(X.data(), sycl::range<1>(X.size()));
        sycl::buffer<double, 1> sum_buf(&sum, sycl::range<1>(1));
        sycl::buffer<double, 1> square_sum_buf(&square_sum, sycl::range<1>(1));
        sycl::buffer<double, 1> cube_sum_buf(&cube_sum, sycl::range<1>(1));
        sycl::buffer<double, 1> min_buf(&min_val, sycl::range<1>(1));
        sycl::buffer<double, 1> max_buf(&max_val, sycl::range<1>(1));
        sycl::buffer<int, 1> bin_buf(bins.data(), sycl::range<1>(bins.size()));

        q.submit([&](sycl::handler& h) {
            auto X_acc = X_buf.get_access<sycl::access::mode::read>(h);
            auto sum_acc = sum_buf.get_access<sycl::access::mode::write>(h);
            auto square_sum_acc = square_sum_buf.get_access<sycl::access::mode::write>(h);
            auto cube_sum_acc = cube_sum_buf.get_access<sycl::access::mode::write>(h);
            auto min_acc = min_buf.get_access<sycl::access::mode::write>(h);
            auto max_acc = max_buf.get_access<sycl::access::mode::write>(h);
            auto bin_acc = bin_buf.get_access<sycl::access::mode::atomic>(h);

            h.parallel_for<class calculate_statistics>(sycl::range<1>(X.size()), [=](sycl::id<1> id) {
                sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> sum_atomic(sum_acc[0]);
                sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> square_sum_atomic(square_sum_acc[0]);
                sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> cube_sum_atomic(cube_sum_acc[0]);
                sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> min_atomic(min_acc[0]);
                sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> max_atomic(max_acc[0]);

                double x = X_acc[id];
                sum_atomic.fetch_add(x);
                square_sum_atomic.fetch_add(x * x);
                cube_sum_atomic.fetch_add(x * x * x);

                double old_min = min_atomic.load(), old_max = max_atomic.load();
                while(x < old_min && !min_atomic.compare_exchange_strong(old_min, x));
                while(x > old_max && !max_atomic.compare_exchange_strong(old_max, x));

            });
        }).wait();
         min_val = min_buf.get_access<sycl::access::mode::read>()[0];
        max_val = max_buf.get_access<sycl::access::mode::read>()[0];

        // Calculate the bin width based on the input range
        bin_width = (max_val - min_val) / num_bins;

        // Submit another kernel to fill the bins
        q.submit([&](sycl::handler& h) {
            auto X_acc = X_buf.get_access<sycl::access::mode::read>(h);
            auto bin_acc = bin_buf.get_access<sycl::access::mode::atomic>(h);

            h.parallel_for<class fill_bins>(sycl::range<1>(X.size()), [=](sycl::id<1> id) {
                double x = X_acc[id];
                int bin_index = static_cast<int>((x - min_val) / bin_width);
                bin_index = max(0, min(bin_index, num_bins - 1));  // Ensure it's within bounds

                bin_acc[bin_index].fetch_add(1, sycl::memory_order::relaxed);
            });
        }).wait();
    }

    double mean = sum / X.size();
    double variance = square_sum / X.size() - mean * mean;
    double skewness = cube_sum / X.size() - 3 * mean * variance - mean * mean * mean;
    skewness /= variance * sqrt(variance);

    // Convert bins to probability density
    vector<double> density(bins.size());
    for (int i = 0; i < bins.size(); ++i) {
        density[i] = static_cast<double>(bins[i]) / X.size();
    }

    return make_tuple(mean, variance, skewness, min_val, max_val, density);
}


int main() {
    sycl::queue q(sycl::default_selector{});

    // Initialize the constants and solution vectors
    vector<double> constants(N+1);
    vector<double> solution_sum(N, 0.0);
    vector<double> average_X(N, 0.0);

    // Initialize the random number generator
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 0.5);
    cout << "Enter the values of a1 to a4: ";
    for (int i = 0; i < 4; i++) {
        cin >> constants[i];
    }

    // Repeat the process for the specified number of iterations
    for (int iter = 0; iter < iterations; ++iter) {
        vector<double> constants_copy(N+1);
        for(int i=0;i<N+1;++i){
            constants_copy[i] = constants[i];
        }

        // Update the constants with the normal distribution
       for (int i = 0; i < N+1; ++i) {
            double x = distribution(generator);
            constants_copy[i] =constants[i]+ x;
           // cout<< "i + x,and x is "<< x <<endl;
             //cout<< "i + x,and i + x is "<< constants_copy[i]  <<endl;
        }

        // Generate the matrix
        vector<vector<double>> matrix = generateMatrix(constants_copy, q);
       // printMatrix(matrix);
        // Create a buffer for the constant vector F
        sycl::buffer<double, 1> F_buf(N);

        // Fill F with ones
        q.submit([&](sycl::handler& h) {
            auto F_accessor = F_buf.get_access<sycl::access::mode::write>(h);
            h.parallel_for<class fillF>(sycl::range<1>(N), [=](sycl::id<1> id) {
                F_accessor[id] = 1.0;
            });
        });

        // Read F buffer into a vector
        vector<double> F(N);
        {
            auto F_acc = F_buf.get_access<sycl::access::mode::read>();
            for (int i = 0; i < N; ++i)
                F[i] = F_acc[i];
        }

        // Solve the equation UX = F
        vector<double> X = solveLinearEquation(q, matrix, F);

        // Accumulate the solution
        for (int i = 0; i < N; ++i) {
            solution_sum[i] += X[i];
            X[i] = 0 ;
        }
    }

    // Calculate and print the average solution
    for (int i = 0; i < N; ++i) {
        average_X[i] = solution_sum[i] / iterations;
        cout << "Average X[" << i << "] = " << average_X[i] << endl;
    }
    double mean, variance, skewness, min_val, max_val;
    vector<double> density;
    tie(mean, variance, skewness, min_val, max_val,density) = calculate_statistics(q, average_X,20,0.1);
    
    cout << "Mean = " << mean << endl;
    cout << "Variance = " << variance << endl;
    cout << "Skewness = " << skewness << endl;
    cout << "Min = " << min_val << endl;
    cout << "Max = " << max_val << endl;
    double bin_width = (max_val - min_val) / 20;
    for (int i = 0; i < density.size(); ++i) {
    double lower = min_val + i * bin_width; // The lower bound of the bin
    double upper = lower + bin_width; // The upper bound of the bin
    cout << "Density[" << i << "] = " << density[i] << " in [" << lower << ", " << upper << ")" << endl;
    }
    return 0;
}