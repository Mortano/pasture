use anyhow::{anyhow, Result};
use std::{cell::RefCell, marker::PhantomData};

use crate::layout::{
    conversion::{get_converter_for_attributes, AttributeConversionFn},
    PointAttributeDefinition, PointAttributeMember, PointType, PrimitiveType,
};

use super::{
    attribute_iterators::{
        AttributeIteratorByMut, AttributeIteratorByRef, AttributeIteratorByValue,
    },
    point_buffer::{
        BorrowedBuffer, BorrowedMutBuffer, ColumnarBuffer, ColumnarBufferMut, InterleavedBuffer,
        InterleavedBufferMut,
    },
    point_iterators::{PointIteratorByMut, PointIteratorByRef, PointIteratorByValue},
    OwningBuffer,
};

/// A strongly typed view over the point data of a buffer. This allows accessing the point data in
/// the buffer using type `T` instead of only through raw memory (i.e. as `&[u8]`). This type makes
/// no assumptions about the memory layout of the underlying buffer, so it only provides access to
/// the point data by value. The `PointView` supports no type conversion, so `T::layout()` must match
/// the `PointLayout` of the buffer. You cannot create instances of `PointView` directly but instead
/// have to use [`BorrowedBuffer::view`] function and its variations, which perform the necessary type
/// checks internally!
///
/// # Lifetime bounds
///
/// Since the `PointView` borrows the buffer internally, and the buffer itself has a borrow lifetime,
/// `PointView` stores two lifetimes so that it can borrow its buffer for a potentially shorter lifetime
/// `'b` than the lifetime `'a` of the buffer itself.
pub struct PointView<'a, 'b, B: BorrowedBuffer<'a>, T: PointType>
where
    'a: 'b,
{
    buffer: &'b B,
    _phantom: PhantomData<&'a T>,
}

impl<'a, 'b, B: BorrowedBuffer<'a>, T: PointType> PointView<'a, 'b, B, T>
where
    'a: 'b,
{
    pub(crate) fn new(buffer: &'b B) -> Self {
        assert_eq!(T::layout(), *buffer.point_layout());
        Self {
            buffer,
            _phantom: Default::default(),
        }
    }

    /// Access the point at `index`
    ///
    /// # Panics
    ///
    /// If `index` is out of bounds
    pub fn at(&self, index: usize) -> T {
        let mut point = T::zeroed();
        self.buffer
            .get_point(index, bytemuck::bytes_of_mut(&mut point));
        point
    }
}

impl<'a, 'b, B: InterleavedBuffer<'a>, T: PointType> PointView<'a, 'b, B, T>
where
    'a: 'b,
{
    /// Access the point at `index` by reference
    ///
    /// # Lifetime bounds
    ///
    /// Just as the `PointView` can borrow its underlying buffer for a shorter lifetime `'b` than
    /// the lifetime `'a` of the buffer, it should be possible to borrow a single point from a `PointView`
    /// for a shorter lifetime `'c` than the lifetime `'b` of the `PointView`, hence the additional
    /// lifetime bounds.
    ///
    /// # Panics
    ///
    /// If `index` is out of bounds
    pub fn at_ref<'c>(&'c self, index: usize) -> &'c T
    where
        'b: 'c,
    {
        bytemuck::from_bytes(self.buffer.get_point_ref(index))
    }

    /// Return an iterator over strongly typed point data by reference
    pub fn iter<'c>(&'c self) -> PointIteratorByRef<'c, T>
    where
        'b: 'c,
    {
        self.buffer.into()
    }
}

impl<'a, 'b, B: BorrowedBuffer<'a> + 'a, T: PointType> IntoIterator for PointView<'a, 'b, B, T>
where
    'a: 'b,
{
    type Item = T;
    type IntoIter = PointIteratorByValue<'a, 'b, T, B>;

    fn into_iter(self) -> Self::IntoIter {
        self.buffer.into()
    }
}

/// Like [`PointView`], but provides mutable access to the strongly typed point data. For buffers with unknown
/// memory layout, this means that you have to use [`PointViewMut::set_at`], but if the underlying buffer
/// implements [`InterleavedBufferMut`], you can also get a mutable borrow the a strongly typed point!
pub struct PointViewMut<'a, 'b, B: BorrowedMutBuffer<'a>, T: PointType>
where
    'a: 'b,
{
    buffer: &'b mut B,
    _phantom: PhantomData<&'a T>,
}

impl<'a, 'b, B: BorrowedMutBuffer<'a>, T: PointType> PointViewMut<'a, 'b, B, T> {
    pub(crate) fn new(buffer: &'b mut B) -> Self {
        assert_eq!(T::layout(), *buffer.point_layout());
        Self {
            buffer,
            _phantom: Default::default(),
        }
    }

    /// Access the point at `index`
    ///
    /// # Panics
    ///
    /// If `index` is out of bounds
    pub fn at(&self, index: usize) -> T {
        let mut point = T::zeroed();
        self.buffer
            .get_point(index, bytemuck::bytes_of_mut(&mut point));
        point
    }

    /// Sets the data for the point at `index`
    ///
    /// # Panics
    ///
    /// If `index` is out of bounds
    pub fn set_at(&mut self, index: usize, point: T) {
        // Safe because `new` asserts that the point layout of `T` and `buffer` match
        unsafe {
            self.buffer.set_point(index, bytemuck::bytes_of(&point));
        }
    }
}

impl<'a, 'b, B: InterleavedBuffer<'a> + BorrowedMutBuffer<'a>, T: PointType>
    PointViewMut<'a, 'b, B, T>
{
    /// Access the point at `index` as an immutable reference
    ///
    /// # Panics
    ///
    /// If `index` is out of bounds
    pub fn at_ref<'c>(&'c self, index: usize) -> &'c T
    where
        'b: 'c,
    {
        bytemuck::from_bytes(self.buffer.get_point_ref(index))
    }

    /// Return an iterator over point data by immutable reference
    pub fn iter<'c>(&'c self) -> PointIteratorByRef<'c, T>
    where
        'b: 'c,
    {
        (&*self.buffer).into()
    }
}

impl<'a, 'b, B: InterleavedBufferMut<'a>, T: PointType> PointViewMut<'a, 'b, B, T> {
    /// Access the point at `index` as a mutable reference
    ///
    /// # Panics
    ///
    /// If `index` is out of bounds
    pub fn at_mut<'c>(&'c mut self, index: usize) -> &'c mut T
    where
        'b: 'c,
    {
        bytemuck::from_bytes_mut(self.buffer.get_point_mut(index))
    }

    /// Returns an iterator over point data by mutable reference
    pub fn iter_mut<'c>(&'c mut self) -> PointIteratorByMut<'c, T>
    where
        'b: 'c,
    {
        self.buffer.into()
    }

    /// Sorts the point buffer using the given `comparator` function
    pub fn sort_by<F: Fn(&T, &T) -> std::cmp::Ordering>(&mut self, comparator: F) {
        let typed_points: &mut [T] =
            bytemuck::cast_slice_mut(self.buffer.get_point_range_mut(0..self.buffer.len()));
        typed_points.sort_by(comparator);
    }
}

impl<'a, 'b, B: OwningBuffer<'a>, T: PointType> PointViewMut<'a, 'b, B, T> {
    /// Push the given `point` into the underlying buffer
    pub fn push_point(&mut self, point: T) {
        // Safe because we know that a `PointViewMut` can never be created for a `T` that is different from
        // the `PointLayout` of the underlying buffer (see the check in `new`)
        unsafe {
            self.buffer.push_points(bytemuck::bytes_of(&point));
        }
    }
}

/// A strongly typed view over attribute data of a point buffer. This allows accessing the data for a specific
/// attribute of a `PointType` using the strong type `T` instead of as raw memory (i.e. `&[u8]`). This type makes
/// no assumptions about the memory layout of the underlying buffer, so it only provides access to the attribute
/// data by value. Just as with the [`PointView`] type, you cannot create instances of `AttributeView` directly.
/// Instead, use the [`BorrowedBuffer::view_attribute`] function and its variations, which perform the necessary
/// type checking.
pub struct AttributeView<'a, 'b, B: BorrowedBuffer<'a>, T: PrimitiveType>
where
    'a: 'b,
{
    buffer: &'b B,
    attribute: &'b PointAttributeMember,
    _phantom: PhantomData<&'a T>,
}

impl<'a, 'b, B: BorrowedBuffer<'a>, T: PrimitiveType> AttributeView<'a, 'b, B, T> {
    pub(crate) fn new(buffer: &'b B, attribute: &PointAttributeDefinition) -> Self {
        assert_eq!(T::data_type(), attribute.datatype());
        Self {
            attribute: buffer
                .point_layout()
                .get_attribute(attribute)
                .expect("Attribute not found in PointLayout of buffer"),
            buffer,
            _phantom: Default::default(),
        }
    }

    /// Get the attribute value at `index`
    ///
    /// # Panics
    ///
    ///  If `index` is out of bounds
    pub fn at(&self, index: usize) -> T {
        let mut attribute = T::zeroed();
        // Is safe because we get the attribute_member from the PointLayout of the buffer in `new`
        unsafe {
            self.buffer.get_attribute_unchecked(
                self.attribute,
                index,
                bytemuck::bytes_of_mut(&mut attribute),
            );
        }
        attribute
    }
}

impl<'a, 'b, B: ColumnarBuffer<'a>, T: PrimitiveType> AttributeView<'a, 'b, B, T>
where
    'a: 'b,
{
    /// Get the attribute value at `index` as an immutable borrow
    ///
    /// # Panics
    ///
    ///  If `index` is out of bounds
    pub fn at_ref<'c>(&'c self, index: usize) -> &'c T
    where
        'b: 'c,
    {
        bytemuck::from_bytes(
            self.buffer
                .get_attribute_ref(self.attribute.attribute_definition(), index),
        )
    }

    /// Returns an iterator over attribute values by immutable reference
    pub fn iter<'c>(&'c self) -> AttributeIteratorByRef<'c, T>
    where
        'b: 'c,
    {
        AttributeIteratorByRef::new(self.buffer, self.attribute.attribute_definition())
    }
}

impl<'a, 'b, B: BorrowedBuffer<'a> + 'a, T: PrimitiveType> IntoIterator
    for AttributeView<'a, 'b, B, T>
{
    type Item = T;
    type IntoIter = AttributeIteratorByValue<'a, 'b, T, B>;

    fn into_iter(self) -> Self::IntoIter {
        AttributeIteratorByValue::new(self.buffer, self.attribute.attribute_definition())
    }
}

/// Like [`AttributeView`], but provides mutable access to the attribute data
pub struct AttributeViewMut<'a, 'b, B: BorrowedMutBuffer<'a>, T: PrimitiveType>
where
    'a: 'b,
{
    buffer: &'b mut B,
    attribute: PointAttributeMember,
    _phantom: PhantomData<&'a T>,
}

impl<'a, 'b, B: BorrowedMutBuffer<'a>, T: PrimitiveType> AttributeViewMut<'a, 'b, B, T>
where
    'a: 'b,
{
    pub(crate) fn new(buffer: &'b mut B, attribute: &PointAttributeDefinition) -> Self {
        assert_eq!(T::data_type(), attribute.datatype());
        Self {
            attribute: buffer
                .point_layout()
                .get_attribute(attribute)
                .expect("Attribute not found in PointLayout of buffer")
                .clone(),
            buffer,
            _phantom: Default::default(),
        }
    }

    /// Get the attribute value at `index`
    ///
    /// # Panics
    ///
    ///  If `index` is out of bounds
    pub fn at(&self, index: usize) -> T {
        let mut attribute = T::zeroed();
        // Is safe because we get the attribute_member from the PointLayout of the buffer in `new`
        unsafe {
            self.buffer.get_attribute_unchecked(
                &self.attribute,
                index,
                bytemuck::bytes_of_mut(&mut attribute),
            );
        }
        attribute
    }

    /// Sets the value of the attribute at `index` to `attribute_value`
    ///
    /// # Panics
    ///
    ///  If `index` is out of bounds
    pub fn set_at(&mut self, index: usize, attribute_value: T) {
        // Safe because `new` checks that the data type of `T` and `self.attribute` match
        unsafe {
            self.buffer.set_attribute(
                self.attribute.attribute_definition(),
                index,
                bytemuck::bytes_of(&attribute_value),
            );
        }
    }
}

impl<'a, 'b, B: ColumnarBuffer<'a> + BorrowedMutBuffer<'a>, T: PrimitiveType>
    AttributeViewMut<'a, 'b, B, T>
where
    'a: 'b,
{
    /// Get the attribute value at `index` as an immutable borrow
    ///
    /// # Panics
    ///
    ///  If `index` is out of bounds
    pub fn at_ref<'c>(&'c self, index: usize) -> &'c T
    where
        'b: 'c,
    {
        bytemuck::from_bytes(
            self.buffer
                .get_attribute_ref(self.attribute.attribute_definition(), index),
        )
    }

    /// Returns an iterator over attribute values as immutable borrows
    pub fn iter<'c>(&'c self) -> AttributeIteratorByRef<'c, T>
    where
        'b: 'c,
    {
        AttributeIteratorByRef::new(self.buffer, self.attribute.attribute_definition())
    }
}

impl<'a, 'b, B: ColumnarBufferMut<'a> + BorrowedMutBuffer<'a>, T: PrimitiveType>
    AttributeViewMut<'a, 'b, B, T>
{
    /// Get the attribute value at `index` as a mutable borrow
    ///
    /// # Panics
    ///
    ///  If `index` is out of bounds
    pub fn at_mut(&'b mut self, index: usize) -> &'b mut T {
        bytemuck::from_bytes_mut(
            self.buffer
                .get_attribute_mut(self.attribute.attribute_definition(), index),
        )
    }

    /// Returns an iterator over attribute values as mutable borrows
    pub fn iter_mut(&'b mut self) -> AttributeIteratorByMut<'b, T> {
        AttributeIteratorByMut::new(self.buffer, self.attribute.attribute_definition())
    }
}

/// A view over a strongly typed point attribute that supports type conversion. This means that the
/// `PointAttributeDataType` of the attribute does not have to match the type `T` that this view returns.
/// For an explanation on how attribute type conversion works in pasture, see the [`conversion`](crate::layout::conversion)
/// module
pub struct AttributeViewConverting<'a, 'b, B: BorrowedBuffer<'a>, T: PrimitiveType>
where
    'a: 'b,
{
    buffer: &'b B,
    attribute: PointAttributeMember,
    converter_fn: AttributeConversionFn,
    converter_buffer: RefCell<Vec<u8>>,
    _phantom: PhantomData<&'a T>,
}

impl<'a, 'b, B: BorrowedBuffer<'a>, T: PrimitiveType> AttributeViewConverting<'a, 'b, B, T> {
    pub(crate) fn new(buffer: &'b B, attribute: &PointAttributeDefinition) -> Result<Self> {
        assert_eq!(T::data_type(), attribute.datatype());
        let attribute_in_layout: &PointAttributeMember = buffer
            .point_layout()
            .get_attribute_by_name(attribute.name())
            .expect("Attribute not found in PointLayout of buffer");
        let converter_fn = get_converter_for_attributes(
            attribute_in_layout.attribute_definition(),
            &attribute.with_custom_datatype(T::data_type()),
        )
        .ok_or(anyhow!("Conversion between attribute types is impossible"))?;
        let converter_buffer = vec![0; T::data_type().size() as usize];
        Ok(Self {
            attribute: attribute_in_layout.clone(),
            buffer,
            converter_fn,
            converter_buffer: RefCell::new(converter_buffer),
            _phantom: Default::default(),
        })
    }

    /// Get the attribute value at `index`
    pub fn at(&self, index: usize) -> T {
        let mut value = T::zeroed();
        // Is safe because we took 'attribute' from the point layout of the buffer
        // conversion is safe because we checked the source and destination types in `new`
        unsafe {
            self.buffer.get_attribute_unchecked(
                &self.attribute,
                index,
                self.converter_buffer.borrow_mut().as_mut_slice(),
            );
            (self.converter_fn)(
                self.converter_buffer.borrow().as_slice(),
                bytemuck::bytes_of_mut(&mut value),
            );
        }
        value
    }
}

impl<'a, 'b, B: BorrowedBuffer<'a>, T: PrimitiveType> IntoIterator
    for AttributeViewConverting<'a, 'b, B, T>
{
    type Item = T;
    type IntoIter = AttributeViewConvertingIterator<'a, 'b, B, T>;

    fn into_iter(self) -> Self::IntoIter {
        AttributeViewConvertingIterator {
            current_index: 0,
            view: self,
        }
    }
}

/// An iterator that performs attribute value conversion on the fly. This allows iterating over an
/// attribute that has internal datatype `U` as if it had datatype `T`
pub struct AttributeViewConvertingIterator<'a, 'b, B: BorrowedBuffer<'a>, T: PrimitiveType> {
    view: AttributeViewConverting<'a, 'b, B, T>,
    current_index: usize,
}

impl<'a, 'b, B: BorrowedBuffer<'a>, T: PrimitiveType> Iterator
    for AttributeViewConvertingIterator<'a, 'b, B, T>
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index == self.view.buffer.len() {
            None
        } else {
            let ret = self.view.at(self.current_index);
            self.current_index += 1;
            Some(ret)
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use super::*;

    use crate::{containers::VectorBuffer, test_utils::*};

    #[test]
    fn test_sort_buffer() {
        let rng = thread_rng();
        let mut test_points = rng
            .sample_iter::<CustomPointTypeSmall, _>(DefaultPointDistribution)
            .take(10)
            .collect::<VectorBuffer>();

        test_points
            .view_mut::<CustomPointTypeSmall>()
            .sort_by(|a, b| a.classification.cmp(&b.classification));

        let points = test_points
            .view::<CustomPointTypeSmall>()
            .into_iter()
            .collect::<Vec<_>>();
        let are_sorted = points
            .iter()
            .zip(points.iter().skip(1))
            .all(|(low, high)| low.classification <= high.classification);
        assert!(are_sorted, "Points not sorted: {:#?}", test_points);
    }
}
