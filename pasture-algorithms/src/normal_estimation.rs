use core::panic;

use kd_tree::{self, KdPoint, KdTree};
use num_traits::{self, Zero};
use pasture_core::layout::{attributes::POSITION_3D, PointType};
use pasture_core::{
    containers::{PerAttributeVecPointStorage, PointBuffer, PointBufferExt},
    nalgebra::{DMatrix, Vector3, Vector4},
};

/// Normal Estimation Algorithm
/// returns a vector of quintupels where each quintupel has the following values: (current point, n_x, n_y, n_z, curvature)
/// iterates over all points in the buffer and constructs new point buffers of size k_nn
/// make sure that knn >= 3 holds as it is not possible to construct a plane over less than three point
///
/// # Examples
///
/// '''
/// # use kd_tree::{self, KdPoint};
/// # use pasture_core::nalgebra::Vector3;
/// # use pasture_core::{containers::InterleavedVecPointStorage, layout::PointType};
/// # use pasture_algorithms::normal_estimation::compute_normals;
/// # use pasture_derive::PointType;
/// # use typenum;

/// #[repr(C)]
/// #[derive(PointType, Debug, Clone, Copy)]
/// struct SimplePoint {
///     #[pasture(BUILTIN_POSITION_3D)]
///     pub position: Vector3<f64>,
///     #[pasture(BUILTIN_INTENSITY)]
///     pub intensity: u16,
/// }
/// impl KdPoint for SimplePoint {
///     type Scalar = f64;
///     type Dim = typenum::U3;
///     fn at(&self, k: usize) -> f64 {
///         return self.position[k];
///     }
/// }
/// fn main() {
///     let points = vec![
///         SimplePoint {
///             position: Vector3::new(11.0, 22.0, 154.0),
///             intensity: 42,
///         },
///         SimplePoint {
///             position: Vector3::new(12.0, 23.0, 0.0),
///             intensity: 84,
///         },
///         SimplePoint {
///             position: Vector3::new(103.0, 84.0, 2.0),
///             intensity: 84,
///         },
///         SimplePoint {
///             position: Vector3::new(101.0, 0.0, 1.0),
///             intensity: 84,
///         },
///     ];

///     let mut interleaved = InterleavedVecPointStorage::new(SimplePoint::layout());

///     interleaved.push_points(points.as_slice());

///     let solution_vec = compute_normals::<InterleavedVecPointStorage, SimplePoint>(&interleaved, 4);
///     for solution in solution_vec {
///        println!(
///             "Point: {:?}, n_x: {}, n_y: {}, n_z: {}, curvature: {}",
///            solution.0, solution.1, solution.2, solution.3, solution.4
///         );
///     }
/// }

pub fn compute_normals<T: PointBuffer, P: PointType + KdPoint + Copy>(
    point_cloud: &T,
    k_nn: usize,
) -> Vec<(P, f64, f64, f64, f64)>
where
    P::Scalar: num_traits::Float,
{
    // this is the solution that will be returned
    let mut points_with_normals_curvature = vec![];

    // transform point cloud in vector of points
    let mut points = vec![];
    for point in point_cloud.iter_point() {
        points.push(point);
    }

    // construct kd tree over the vector of points.
    let cloud_as_kd_tree: KdTree<P> = KdTree::build_by_ordered_float(points);

    // iterate over all points in the point cloud and and calculate the k nearest neighbors with the constructed kd tree
    for point in point_cloud.iter_point::<P>() {
        let nearest_points = cloud_as_kd_tree
            .nearests(&point, k_nn)
            .iter()
            .map(|x| x.item)
            .collect::<Vec<&P>>();

        // stores the k nearest neighbors in a PointStorage
        let mut k_nn_buffer = PerAttributeVecPointStorage::new(point_cloud.point_layout().clone());
        for point in nearest_points {
            k_nn_buffer.push_point(*point);
        }

        // coordinates of the surface normal
        let mut n_x: f64 = 0.0;
        let mut n_y: f64 = 0.0;
        let mut n_z: f64 = 0.0;

        // curvature
        let mut curvature: f64 = 0.0;

        normal_estimation(&k_nn_buffer, &mut n_x, &mut n_y, &mut n_z, &mut curvature);

        // solution vector
        points_with_normals_curvature.push((point, n_x, n_y, n_z, curvature));
    }

    return points_with_normals_curvature;
}

/// checks whether a given point cloud has points with coordinates that are Not a Number
fn is_dense<T: PointBuffer>(point_cloud: &T) -> bool {
    for point in point_cloud.iter_attribute::<Vector3<f64>>(&POSITION_3D) {
        if point.x.is_nan() || point.y.is_nan() || point.z.is_nan() {
            return false;
        }
    }
    return true;
}

/// checks whether a given point has finite coordinates
fn is_finite(point: &Vector3<f64>) -> bool {
    if !(point.x.is_finite() || point.y.is_finite() || point.z.is_finite()) {
        return false;
    }
    return true;
}

/// computes the centroid for a given point cloud
/// the centroid is the point that has the same distance to all other points in the point cloud
fn compute_centroid<T: PointBuffer>(point_cloud: &T, centroid: &mut Vector4<f64>) {
    if point_cloud.is_empty() {
        panic!("The point cloud is empty!");
    }

    centroid.set_zero();
    let mut temp_centroid = vec![0.0; 3];

    if is_dense(point_cloud) {
        // add all points up
        for point in point_cloud.iter_attribute::<Vector3<f64>>(&POSITION_3D) {
            temp_centroid[0] += point.x;
            temp_centroid[1] += point.y;
            temp_centroid[2] += point.z;
        }

        //normalize over all points
        centroid[0] = temp_centroid[0] / point_cloud.len() as f64;
        centroid[1] = temp_centroid[1] / point_cloud.len() as f64;
        centroid[2] = temp_centroid[2] / point_cloud.len() as f64;
        centroid[3] = 1.0;
    } else {
        let mut points_in_cloud = 0;
        for point in point_cloud.iter_attribute::<Vector3<f64>>(&POSITION_3D) {
            if is_finite(&point) {
                // add all points up
                temp_centroid[0] += point.x;
                temp_centroid[1] += point.y;
                temp_centroid[2] += point.z;
                points_in_cloud = points_in_cloud + 1;
            }
        }

        // normalize over all points
        centroid[0] = temp_centroid[0] / points_in_cloud as f64;
        centroid[1] = temp_centroid[1] / points_in_cloud as f64;
        centroid[2] = temp_centroid[2] / points_in_cloud as f64;
        centroid[3] = 1.0;
    }
}

/// compute the covariance matrix for a given point cloud which is a measure of spread out the points are
fn compute_covarianz_matrix<T: PointBuffer>(
    point_cloud: &T,
    covariance_matrix: &mut DMatrix<f64>,
) -> usize {
    let mut point_count = 0;
    if point_cloud.is_empty() {
        return 0;
    }
    // compute the centroid of the point cloud
    let mut centroid = Vector4::<f64>::zeros();
    compute_centroid(point_cloud, &mut centroid);

    if is_dense(point_cloud) {
        point_count = point_cloud.len();
        let mut diff_mean = vec![0.0; 4];
        for point in point_cloud.iter_attribute::<Vector3<f64>>(&POSITION_3D) {
            // calculate difference from the centroid for each point
            diff_mean[0] = point.x - centroid[0];
            diff_mean[1] = point.y - centroid[1];
            diff_mean[2] = point.z - centroid[2];

            covariance_matrix[(1, 1)] += diff_mean[1] * diff_mean[1];
            covariance_matrix[(1, 2)] += diff_mean[1] * diff_mean[2];
            covariance_matrix[(2, 2)] += diff_mean[2] * diff_mean[2];

            let diff_x = diff_mean[0];
            diff_mean.iter_mut().for_each(|x| *x = *x * diff_x);

            covariance_matrix[(0, 0)] += diff_mean[0];
            covariance_matrix[(0, 1)] += diff_mean[1];
            covariance_matrix[(0, 2)] += diff_mean[2];
        }
    } else {
        // in this case we dont know the number of points in the point cloud that are finite
        for point in point_cloud.iter_attribute::<Vector3<f64>>(&POSITION_3D) {
            if !is_finite(&point) {
                continue;
            }
            // only compute the covariance matrix for finite points
            let mut diff_mean = vec![0.0; 4];
            // calculate difference from the centroid for each point
            diff_mean[0] = point.x - centroid[0];
            diff_mean[1] = point.y - centroid[1];
            diff_mean[2] = point.z - centroid[2];

            covariance_matrix[(1, 1)] += diff_mean[1] * diff_mean[1];
            covariance_matrix[(1, 2)] += diff_mean[1] * diff_mean[2];
            covariance_matrix[(2, 2)] += diff_mean[2] * diff_mean[2];

            let diff_x = diff_mean[0];
            diff_mean.iter_mut().for_each(|x| *x = *x * diff_x);

            covariance_matrix[(0, 0)] += diff_mean[0];
            covariance_matrix[(0, 1)] += diff_mean[1];
            covariance_matrix[(0, 2)] += diff_mean[2];
            point_count = point_count + 1;
        }
    }

    covariance_matrix[(1, 0)] = covariance_matrix[(0, 1)];
    covariance_matrix[(2, 0)] = covariance_matrix[(0, 2)];
    covariance_matrix[(2, 1)] = covariance_matrix[(1, 2)];

    return point_count;
}

/// find the eigen value solution if the highest degree of the polynomial is 2
fn solve_polynomial_quadratic(
    coefficient_2: &f64,
    coefficient_1: &f64,
    eigen_values: &mut Vec<f64>,
) {
    eigen_values[0] = 0.0;

    let mut delta = coefficient_2 * coefficient_2 - 4.0 * coefficient_1;

    if delta < 0.0 {
        delta = 0.0;
    }

    let sqrt_delta = f64::sqrt(delta);

    eigen_values[2] = 0.5 * (coefficient_2 + sqrt_delta);
    eigen_values[1] = 0.5 * (coefficient_2 - sqrt_delta);
}

/// solve the polynomial to find the eigen values for a given covariance matrix
fn solve_polynomial(covariance_matrix: &DMatrix<f64>, eigen_values: &mut Vec<f64>) {
    let coefficient_0 = covariance_matrix[(0, 0)]
        * covariance_matrix[(1, 1)]
        * covariance_matrix[(2, 2)]
        + 2.0 * covariance_matrix[(0, 1)] * covariance_matrix[(0, 2)] * covariance_matrix[(1, 2)]
        - covariance_matrix[(0, 0)] * covariance_matrix[(1, 2)] * covariance_matrix[(1, 2)]
        - covariance_matrix[(1, 1)] * covariance_matrix[(0, 2)] * covariance_matrix[(0, 2)]
        - covariance_matrix[(2, 2)] * covariance_matrix[(0, 1)] * covariance_matrix[(0, 1)];
    let coefficient_1 = covariance_matrix[(0, 0)] * covariance_matrix[(1, 1)]
        - covariance_matrix[(0, 1)] * covariance_matrix[(0, 1)]
        + covariance_matrix[(0, 0)] * covariance_matrix[(2, 2)]
        - covariance_matrix[(0, 2)] * covariance_matrix[(0, 2)]
        + covariance_matrix[(1, 1)] * covariance_matrix[(2, 2)]
        - covariance_matrix[(1, 2)] * covariance_matrix[(1, 2)];
    let coefficient_2 =
        covariance_matrix[(0, 0)] + covariance_matrix[(1, 1)] + covariance_matrix[(2, 2)];

    // check if one eigen value solution is zero
    if coefficient_0.abs() < std::f64::EPSILON {
        solve_polynomial_quadratic(&coefficient_2, &coefficient_1, eigen_values);
    } else {
        let one_third = 1.0 / 3.0;
        let sqrt_3 = f64::sqrt(3.0);

        let coefficient_2_third = coefficient_2 * one_third;
        let mut alpha_third = (coefficient_1 - coefficient_2 * coefficient_2_third) * one_third;
        if alpha_third > 0.0 {
            alpha_third = 0.0;
        }

        let half_beta = 0.5
            * (coefficient_0
                + coefficient_2_third
                    * (2.0 * coefficient_2_third * coefficient_2_third - coefficient_1));

        let mut q = half_beta * half_beta + alpha_third * alpha_third * alpha_third;
        if q > 0.0 {
            q = 0.0;
        }

        // calculate eigen values
        let rho = f64::sqrt(-alpha_third);
        let theta = f64::atan2(f64::sqrt(-q), half_beta) * one_third;
        let cosine_of_theta = f64::cos(theta);
        let sine_of_theta = f64::sin(theta);

        eigen_values[0] = coefficient_2_third + 2.0 * rho * cosine_of_theta;
        eigen_values[1] = coefficient_2_third - rho * (cosine_of_theta + sqrt_3 * sine_of_theta);
        eigen_values[2] = coefficient_2_third - rho * (cosine_of_theta - sqrt_3 * sine_of_theta);

        // sort increasing so that eigen_values[0] is the smallest eigen value
        eigen_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // if the smallest eigen value is zero or less the solution is a quadratic
        if eigen_values[0] <= 0.0 {
            solve_polynomial_quadratic(&coefficient_2, &coefficient_1, eigen_values);
        }
    }
}

/// calculates the largest eigen vector for a given matrix
fn get_largest_eigen_vector(scaled_matrix: &DMatrix<f64>, eigen_vector: &mut Vec<f64>) {
    let mut rows = vec![];
    rows.push(scaled_matrix.row(0).cross(&scaled_matrix.row(1)));
    rows.push(scaled_matrix.row(0).cross(&scaled_matrix.row(2)));
    rows.push(scaled_matrix.row(1).cross(&scaled_matrix.row(2)));

    let mut cross_product = DMatrix::<f64>::zeros(3, 3);

    // write rows of cross product
    for it in cross_product.row_iter_mut().zip(rows) {
        let (mut cross_row, row) = it;
        // row from rows vector is written to the row of the cross product
        cross_row.copy_from(&row);
    }

    // find largest eigen vector
    let mut largest_eigen_vec = cross_product.row(0);
    for row in cross_product.row_iter() {
        if row.norm() > largest_eigen_vec.norm() {
            largest_eigen_vec = row;
        }
    }

    // set eigen vector to largest vector
    for i in 0..eigen_vector.len() {
        eigen_vector[i] = largest_eigen_vec[i];
    }
}

/// for a given 3x3 matrix the functions calculates the eigen vector of the smallest eigen value that can be found
fn eigen_3x3(covariance_matrix: &DMatrix<f64>, eigen_value: &mut f64, eigen_vector: &mut Vec<f64>) {
    let scale = covariance_matrix.abs().max();
    let mut covariance_matrix_scaled = DMatrix::<f64>::zeros(3, 3);
    for i in 0..covariance_matrix.len() {
        covariance_matrix_scaled[i] = covariance_matrix[i] / scale;
    }

    // scale the matrix down
    for (index, value) in covariance_matrix.iter().enumerate() {
        covariance_matrix_scaled[index] = value / scale;
    }

    let mut eigen_values = vec![0.0; 3];
    solve_polynomial(covariance_matrix, &mut eigen_values);
    // undo scale for smallest eigen vector
    *eigen_value = eigen_values[0] * scale;

    // subtract the smallest eigen value from the diagonal of the matrix
    covariance_matrix_scaled
        .diagonal()
        .iter_mut()
        .for_each(|x| *x = *x - eigen_values[0]);

    get_largest_eigen_vector(&mut covariance_matrix_scaled, eigen_vector);
}

/// calculates the orientation of the surface as a normal vector with components n_x, n_y and n_z as well as the curvature of the surface for a given covariance matrix
fn solve_plane_parameter(
    covariance_matrix: &DMatrix<f64>,
    eigen_vector: &mut Vec<f64>,
    curvature: &mut f64,
) {
    let mut eigen_value: f64 = 0.0;
    eigen_3x3(covariance_matrix, &mut eigen_value, eigen_vector);

    let eigen_sum = covariance_matrix[0] + covariance_matrix[4] + covariance_matrix[8];
    if eigen_sum != 0.0 {
        *curvature = (eigen_value / eigen_sum).abs();
    } else {
        *curvature = 0.0;
    }
}

/// calculates the normal vectors and the curvature of the surface for the given point cloud
fn normal_estimation<T: PointBuffer>(
    //this needs to be k nearest neighbors
    point_cloud: &T,
    n_x: &mut f64,
    n_y: &mut f64,
    n_z: &mut f64,
    curvature: &mut f64,
) {
    let mut covariance_matrix = DMatrix::<f64>::zeros(3, 3);

    if point_cloud.len() < 3 || compute_covarianz_matrix(point_cloud, &mut covariance_matrix) == 0 {
        *n_x = f64::NAN;
        *n_y = f64::NAN;
        *n_z = f64::NAN;
        *curvature = f64::NAN;
        return;
    }

    let mut eigen_vector = vec![0.0; 3];
    solve_plane_parameter(&covariance_matrix, &mut eigen_vector, curvature);

    *n_x = eigen_vector[0];
    *n_y = eigen_vector[1];
    *n_z = eigen_vector[2];
}

#[cfg(test)]
mod tests {

    use pasture_core::{
        containers::InterleavedVecPointStorage, layout::PointType, nalgebra::Matrix3,
        nalgebra::Vector3,
    };
    use pasture_derive::PointType;

    use super::*;

    #[repr(C)]
    #[derive(PointType, Debug, Clone, Copy)]
    pub struct SimplePoint {
        #[pasture(BUILTIN_POSITION_3D)]
        pub position: Vector3<f64>,
        #[pasture(BUILTIN_INTENSITY)]
        pub intensity: u16,
    }
    impl KdPoint for SimplePoint {
        type Scalar = f64;
        type Dim = typenum::U3;
        fn at(&self, k: usize) -> f64 {
            return self.position[k];
        }
    }

    #[test]
    fn test_compute_normal_sub() {
        let points: Vec<SimplePoint> = vec![
            SimplePoint {
                position: Vector3::new(1.0, 0.0, 0.0),
                intensity: 42,
            },
            SimplePoint {
                position: Vector3::new(0.0, 1.0, 0.0),
                intensity: 84,
            },
            SimplePoint {
                position: Vector3::new(1.0, 1.0, 0.0),
                intensity: 84,
            },
            SimplePoint {
                position: Vector3::new(-1.0, 0.0, 0.0),
                intensity: 84,
            },
        ];

        let mut interleaved = InterleavedVecPointStorage::new(SimplePoint::layout());

        interleaved.push_points(points.as_slice());

        let mut centroid = Vector4::<f64>::new_random();
        compute_centroid(&interleaved, &mut centroid);
        let result_centroid = Vector4::new(0.25, 0.5, 0.0, 1.0);
        assert_eq!(centroid, result_centroid);

        let mut covariance_matrix = DMatrix::<f64>::zeros(3, 3);
        compute_covarianz_matrix(&interleaved, &mut covariance_matrix);
        let result = Matrix3::new(
            0.6875 * 4.0,
            0.125 * 4.0,
            0.0,
            0.125 * 4.0,
            0.25 * 4.0,
            0.0,
            0.0,
            0.0,
            0.0,
        );
        assert_eq!(covariance_matrix, result);

        let mut normal_vec = vec![0.0; 3];
        let mut curvature: f64 = 0.0;

        solve_plane_parameter(&covariance_matrix, &mut normal_vec, &mut curvature);

        assert_eq!(normal_vec[0], 0.0);
        assert_eq!(normal_vec[1], 0.0);
        assert_ne!(normal_vec[2], 0.0);
        assert_eq!(curvature, 0.0);
    }

    #[test]
    fn test_compute_normal() {
        let points: Vec<SimplePoint> = vec![
            SimplePoint {
                position: Vector3::new(1.0, 0.0, 0.0),
                intensity: 42,
            },
            SimplePoint {
                position: Vector3::new(0.0, 1.0, 0.0),
                intensity: 84,
            },
            SimplePoint {
                position: Vector3::new(1.0, 1.0, 0.0),
                intensity: 84,
            },
            SimplePoint {
                position: Vector3::new(-1.0, 0.0, 0.0),
                intensity: 84,
            },
        ];

        let mut interleaved = InterleavedVecPointStorage::new(SimplePoint::layout());

        interleaved.push_points(points.as_slice());

        let solution_vec =
            compute_normals::<InterleavedVecPointStorage, SimplePoint>(&interleaved, 3);
        for solution in solution_vec {
            assert_eq!(solution.1, 0.0);
            assert_eq!(solution.2, 0.0);
            assert_ne!(solution.3, 0.0);
            assert_eq!(solution.4, 0.0);
        }
    }
}
