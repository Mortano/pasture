#![warn(clippy::all)]
//! Algorithms that operate on point-buffers.
//!
//! Pasture contains algorithms that can manipulate the point cloud data or
//! calculate results based on them.

// Algorithm to calculate the bounding box of a point cloud.
pub mod bounds;
// Get the minimum and maximum value of a specific attribute in a point cloud.
pub mod minmax;
// Contains ransac line- and plane-segmentation algorithms in serial and parallel that can be used
// to get the best line-/plane-model and the corresponding inlier indices.
pub mod segmentation;
// Contains a normal estimation algorithm that can be used to determine the orientation of the surface
// over a point and its k nearest neighbors. The algorithm also determine the curvature of the surface
pub mod normal_estimation;

