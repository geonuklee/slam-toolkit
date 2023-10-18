#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <cstdlib>  // for srand
#include <omp.h>
#if 0
Eigen::Matrix4f Estimate3D3DRANSAC( const std::vector<cv::Point3f>& src, 
    const std::vector<cv::Point3f>& dst, 
    int iterations, 
    float threshold) {
    assert(src.size() == dst.size() && "Source and destination point vectors must be of equal size");

    int n = src.size();
    Eigen::Matrix4f bestTcw = Eigen::Matrix4f::Identity();

    // Declare shared variables
    int bestInliersCount = 0;
    Eigen::Matrix4f TcwShared;

    // Begin parallel region
    #pragma omp parallel private(TcwShared)
    {
        // Declare private variables
        int idx1, idx2, idx3;
        Eigen::Matrix3f srcMat, dstMat;

        // Begin parallel for loop
        #pragma omp for reduction(max:bestInliersCount)
        for (int i = 0; i < iterations; ++i) {
            // Randomly select 3 non-collinear points
            idx1 = rand_r() % n;
            idx2 = rand_r() % n;
            idx3 = rand_r() % n;

            // Ensure the points are not collinear and not the same
            if (idx1 != idx2 && idx1 != idx3 && idx2 != idx3) {
              // && !collinear(src[idx1], src[idx2], src[idx3]) && !collinear(dst[idx1], dst[idx2], dst[idx3])) {
                // Create matrices for the points
                srcMat << src[idx1].x, src[idx1].y, src[idx1].z,
                          src[idx2].x, src[idx2].y, src[idx2].z,
                          src[idx3].x, src[idx3].y, src[idx3].z;
                dstMat << dst[idx1].x, dst[idx1].y, dst[idx1].z,
                          dst[idx2].x, dst[idx2].y, dst[idx2].z,
                          dst[idx3].x, dst[idx3].y, dst[idx3].z;

                // Compute the transformation matrix
                TcwShared = Eigen::Matrix4f::Identity();
                TcwShared.block<3,3>(0, 0) = srcMat.inverse() * dstMat;

                // Compute the error for all points
                int inliersCount = 0;
                for (int j = 0; j < n; ++j) {
                    Eigen::Vector4f srcHomogeneous(src[j].x, src[j].y, src[j].z, 1.0f);
                    Eigen::Vector4f dstHomogeneous(dst[j].x, dst[j].y, dst[j].z, 1.0f);

                    Eigen::Vector4f error = TcwShared * srcHomogeneous - dstHomogeneous;
                    if (error.norm() < threshold) {
                        ++inliersCount;
                    }
                }

                // Update the best transformation matrix if more inliers are found
                if (inliersCount > bestInliersCount) {
                    bestInliersCount = inliersCount;
                    bestTcw = TcwShared;
                }
            }
        }
    }

    return bestTcw;
}
#endif
