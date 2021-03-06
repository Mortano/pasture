use pasture_core::{
    containers::{
        InterleavedPointBufferMutExt, PerAttributePointBufferMutExt, PerAttributeVecPointStorage,
        PointBufferExt,
    },
    nalgebra::Vector3,
};
use pasture_core::{
    containers::{InterleavedVecPointStorage, PerAttributePointBuffer},
    layout::{
        attributes::{INTENSITY, POSITION_3D},
        PointType,
    },
};
use pasture_derive::PointType;

/// We define a simple point type here that has two attributes: 3D position and intensity
#[repr(C)]
#[derive(PointType, Debug)]
struct SimplePoint {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
}

fn main() {
    // Create some points
    let points = vec![
        SimplePoint {
            position: Vector3::new(1.0, 2.0, 3.0),
            intensity: 42,
        },
        SimplePoint {
            position: Vector3::new(-1.0, -2.0, -3.0),
            intensity: 84,
        },
    ];

    // By default, our data is in interleaved format, because a struct is a form of interleaved data. So
    // let's create a buffer to hold our points:
    {
        let mut buffer = InterleavedVecPointStorage::new(SimplePoint::layout());
        // We can add interleaved data like so:
        buffer.push_points(points.as_slice());

        println!("Iterating over interleaved points:");
        // The buffer itself is not strongly typed, but there are some helper methods in the `PointBufferExt` trait to access the data in
        // a strongly typed fashion. `iter_point<T>` creates an iterator over strongly typed points in the buffer:
        for point in buffer.iter_point::<SimplePoint>() {
            println!("{:?}", point);
        }

        // The iterator returned by `iter_point<T>` iterates over the points by value. Let's try mutating the points instead. For this, we
        // can use the `InterleavedPointBufferMutExt` trait. `iter_point_mut<T>` creates an iterator over strongly typed mutable references
        // to the points in the buffer:
        for point_mut in buffer.iter_point_mut::<SimplePoint>() {
            point_mut.intensity *= 2;
        }

        // We can also directly slice our buffer (also see the docs of the `slice` method which explains the syntax)
        println!("Iterating over interleaved points slice:");
        let sliced = buffer.slice(1..2);
        for point in sliced.iter_point::<SimplePoint>() {
            println!("{:?}", point);
        }
    }

    // There are several different types of point buffers. Most code in Pasture can deal with any of these buffer types, though
    // sometimes this is not possible due to memory layout concerns or general performance.
    // Let's try a different type of buffer:
    {
        let mut buffer = PerAttributeVecPointStorage::new(SimplePoint::layout());
        // This buffer stores points with a different memory layout internally (PerAttribute as opposed to Interleaved). We can
        // still add our strongly typed points to it:
        buffer.push_points(points.as_slice());

        //... and iterate it:
        println!("Iterating over per-attribute points:");
        for point in buffer.iter_point::<SimplePoint>() {
            println!("{:?}", point);
        }

        // With the PerAttribute memory layout, we can iterate over specific attributes and even mutate them, instead of always
        // iterating over the whole point. This can give better performance in many cases.
        // As the buffer is not strongly typed, we need to specify the type of the attribute, similar to the call to `iter_point<T>`
        // before. In addition, we have to give Pasture an 'attribute specifier' to determine which attribute we want:
        println!("Iterating over a single attribute:");
        for position in buffer.iter_attribute::<Vector3<f64>>(&POSITION_3D) {
            // Notice that `iter_attribute<T>` returns `T` by value. It is available for all point buffer types, at the expense of
            // only receiving a copy of the attribute.
            println!("Position: {:?}", position);
        }

        // There are several builtin attribute specifiers in the namespace `pasture_core::layout::attributes`.These are the ones that
        // are used when you `#[derive(PointType)]` and say `#[pasture(BUILTIN_XYZ)]`. An attribute specifier internally uses a unique
        // name to identify the attribute, as well as the default datatype of the attribute. Using the builtin specifiers guarantees that
        // all attributes are always correctly addressed.

        // Let's try mutating a specific attribute. This is only possible for a buffer that stores data in PerAttribute memory layout. We
        // can use the `PerAttributePointBufferMutExt` extension trait, which gives us a method to obtain an iterator over mutable references
        // to attribute values:
        for intensity in buffer.iter_attribute_mut::<u16>(&INTENSITY) {
            *intensity *= 2;
        }

        // Just as with the Interleaved buffer, we can slice (but make sure the `PerAttributePointBuffer` trait is in scope!):
        println!("Iterating over per-attribute point slice:");
        let sliced = buffer.slice(1..2);
        for point in sliced.iter_point::<SimplePoint>() {
            println!("{:?}", point);
        }
    }
}
