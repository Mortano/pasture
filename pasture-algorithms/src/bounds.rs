use pasture_core::{
    containers::{PointBuffer, PointBufferExt},
    layout::attributes::POSITION_3D,
    math::AABB,
    nalgebra::{Point3, Vector3},
};

/// Calculate the bounding box of the points in the given `buffer`. Returns `None` if the buffer contains zero
/// points, or if the `PointLayout` of the buffer does not contain the `POSITION_3D` attribute
pub fn calculate_bounds<T: PointBuffer>(buffer: &T) -> Option<AABB<f64>> {
    if buffer.len() == 0 {
        return None;
    }
    let position_attribute = match buffer
        .point_layout()
        .get_attribute_by_name(POSITION_3D.name())
    {
        Some(a) => a,
        None => return None,
    };

    if position_attribute.datatype() == POSITION_3D.datatype() {
        Some(calculate_bounds_from_default_positions(buffer))
    } else {
        Some(calculate_bounds_from_custom_positions(buffer))
    }
}

fn calculate_bounds_from_default_positions<T: PointBuffer>(buffer: &T) -> AABB<f64> {
    let mut pos_min = Point3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut pos_max = Point3::new(f64::MIN, f64::MIN, f64::MIN);
    for pos in buffer.iter_attribute::<Vector3<f64>>(&POSITION_3D) {
        if pos.x < pos_min.x {
            pos_min.x = pos.x;
        }
        if pos.y < pos_min.y {
            pos_min.y = pos.y;
        }
        if pos.z < pos_min.z {
            pos_min.z = pos.z;
        }
        if pos.x > pos_max.x {
            pos_max.x = pos.x;
        }
        if pos.y > pos_max.y {
            pos_max.y = pos.y;
        }
        if pos.z > pos_max.z {
            pos_max.z = pos.z;
        }
    }
    AABB::from_min_max(pos_min, pos_max)
}

fn calculate_bounds_from_custom_positions<T: PointBuffer>(buffer: &T) -> AABB<f64> {
    let mut pos_min = Point3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut pos_max = Point3::new(f64::MIN, f64::MIN, f64::MIN);
    for pos in buffer.iter_attribute_as::<Vector3<f64>>(&POSITION_3D) {
        if pos.x < pos_min.x {
            pos_min.x = pos.x;
        }
        if pos.y < pos_min.y {
            pos_min.y = pos.y;
        }
        if pos.z < pos_min.z {
            pos_min.z = pos.z;
        }
        if pos.x > pos_max.x {
            pos_max.x = pos.x;
        }
        if pos.y > pos_max.y {
            pos_max.y = pos.y;
        }
        if pos.z > pos_max.z {
            pos_max.z = pos.z;
        }
    }
    AABB::from_min_max(pos_min, pos_max)
}
