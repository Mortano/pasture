use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use anyhow::{anyhow, Context, Result};
use pasture_core::layout::PointLayout;

// use crate::las::{LASReader, LASWriter};

use crate::{
    las::{LASReader, LASWriter},
    tiles3d::{PntsReader, PntsWriter},
};

use super::{PointReader, PointWriter};

#[derive(Debug)]
enum SupportedFileExtensions {
    LAS,
    Tiles3D,
}

/// Returns a lookup value for the file extension of the given file path
fn get_extension_lookup(path: &Path) -> Result<SupportedFileExtensions> {
    let extension = path.extension().ok_or_else(|| {
        anyhow!(
            "File extension could not be determined from path {}",
            path.display()
        )
    })?;
    let extension_str = extension.to_str().ok_or_else(|| {
        anyhow!(
            "File extension of path {} is no valid Unicode string",
            path.display()
        )
    })?;
    match extension_str.to_lowercase().as_str() {
        "las" | "laz" => Ok(SupportedFileExtensions::LAS),
        "pnts" => Ok(SupportedFileExtensions::Tiles3D),
        other => Err(anyhow!("Unsupported file extension {other}")),
    }
}

pub enum GenericPointReader {
    LAS(LASReader<'static, BufReader<File>>),
    Tiles3D(PntsReader<BufReader<File>>),
}

impl GenericPointReader {
    pub fn open_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let extension = get_extension_lookup(path.as_ref())?;
        match extension {
            SupportedFileExtensions::LAS => {
                let reader = LASReader::from_path(path)?;
                Ok(Self::LAS(reader))
            }
            SupportedFileExtensions::Tiles3D => {
                let reader = PntsReader::from_path(path)?;
                Ok(Self::Tiles3D(reader))
            }
        }
    }

    /// Returns the total number of points in the underlying point cloud file. Returns `None` if the number of
    /// points is unknown (e.g. for ASCII files which don't have header information)
    pub fn point_count(&self) -> Option<usize> {
        match self {
            GenericPointReader::LAS(reader) => reader.get_metadata().number_of_points(),
            GenericPointReader::Tiles3D(reader) => reader.get_metadata().number_of_points(),
        }
    }
}

impl PointReader for GenericPointReader {
    fn read_into<'a, 'b, B: pasture_core::containers::OwningBuffer<'a>>(
        &mut self,
        point_buffer: &'b mut B,
        count: usize,
    ) -> Result<usize>
    where
        'a: 'b,
    {
        match self {
            GenericPointReader::LAS(reader) => reader.read_into(point_buffer, count),
            GenericPointReader::Tiles3D(reader) => reader.read_into(point_buffer, count),
        }
    }

    fn get_metadata(&self) -> &dyn pasture_core::meta::Metadata {
        match self {
            GenericPointReader::LAS(reader) => reader.get_metadata(),
            GenericPointReader::Tiles3D(reader) => reader.get_metadata(),
        }
    }

    fn get_default_point_layout(&self) -> &PointLayout {
        match self {
            GenericPointReader::LAS(reader) => reader.get_default_point_layout(),
            GenericPointReader::Tiles3D(reader) => reader.get_default_point_layout(),
        }
    }
}

pub enum GenericPointWriter {
    LAS(LASWriter<BufWriter<File>>),
    Tiles3D(PntsWriter<BufWriter<File>>),
}

impl GenericPointWriter {
    pub fn open_file<P: AsRef<Path>>(path: P, point_layout: &PointLayout) -> Result<Self> {
        let extension = get_extension_lookup(path.as_ref())?;
        match extension {
            SupportedFileExtensions::LAS => {
                let writer = LASWriter::from_path_and_point_layout(path, point_layout)?;
                Ok(Self::LAS(writer))
            }
            SupportedFileExtensions::Tiles3D => {
                let file = BufWriter::new(File::create(path.as_ref()).context(format!(
                    "Could not open file {} for writing",
                    path.as_ref().display()
                ))?);
                let writer = PntsWriter::from_write_and_layout(file, point_layout.clone());
                Ok(Self::Tiles3D(writer))
            }
        }
    }
}

impl PointWriter for GenericPointWriter {
    fn write<'a, B: pasture_core::containers::BorrowedBuffer<'a>>(
        &mut self,
        points: &'a B,
    ) -> Result<()> {
        match self {
            GenericPointWriter::LAS(writer) => writer.write(points),
            GenericPointWriter::Tiles3D(writer) => writer.write(points),
        }
    }

    fn flush(&mut self) -> Result<()> {
        match self {
            GenericPointWriter::LAS(writer) => writer.flush(),
            GenericPointWriter::Tiles3D(writer) => writer.flush(),
        }
    }

    fn get_default_point_layout(&self) -> &PointLayout {
        match self {
            GenericPointWriter::LAS(writer) => writer.get_default_point_layout(),
            GenericPointWriter::Tiles3D(writer) => writer.get_default_point_layout(),
        }
    }
}

// type ReaderFactoryFn = dyn Fn(&Path) -> Result<GenericPointReader>;
// type WriterFactoryFn = dyn Fn(&Path, &PointLayout) -> Result<GenericPointWriter>;

// /// Factory that can create `PointReader` and `PointWriter` objects based on file extensions. Use this if you have a file path
// /// and just want to create a `PointReader` or `PointWriter` from this path, without knowing the type of file. The `Default`
// /// implementation supports all file formats that Pasture natively works with, custom formats can be registered using the
// /// `register_...` functions. An extension in this context is whatever [`Path::extension()`](Path::extension) returns for a valid file path
// pub struct IOFactory {
//     reader_factories: HashMap<String, Box<ReaderFactoryFn>>,
//     writer_factories: HashMap<String, Box<WriterFactoryFn>>,
// }

// impl IOFactory {
//     /// Try to create a `PointReader` that can read from the given `file`. This function will fail if `file` has
//     /// a format that is unsupported by Pasture, or if there are any I/O errors while trying to access `file`.
//     pub fn make_reader(&self, file: &Path) -> Result<GenericPointReader> {
//         let extension = file.extension().ok_or_else(|| {
//             anyhow!(
//                 "File extension could not be determined from path {}",
//                 file.display()
//             )
//         })?;
//         let extension_str = extension.to_str().ok_or_else(|| {
//             anyhow!(
//                 "File extension of path {} is no valid Unicode string",
//                 file.display()
//             )
//         })?;
//         let extension_str_lower = extension_str.to_lowercase();
//         let factory = self
//             .reader_factories
//             .get(extension_str_lower.as_str())
//             .ok_or_else(|| {
//                 anyhow!(
//                     "Reading from point cloud files with extension {} is not supported",
//                     extension_str
//                 )
//             })?;

//         factory(file)
//     }

//     /// Try to create a `PointWriter` for writing into the given `file` with the given `point_layout`. This function
//     /// will fail if `file` has a format that is unsupported by pasture, or if there are any I/O errors while trying
//     /// to access `file`.
//     pub fn make_writer(
//         &self,
//         file: &Path,
//         point_layout: &PointLayout,
//     ) -> Result<Box<dyn PointWriter>> {
//         let extension = file.extension().ok_or_else(|| {
//             anyhow!(
//                 "File extension could not be determined from path {}",
//                 file.display()
//             )
//         })?;
//         let extension_str = extension.to_str().ok_or_else(|| {
//             anyhow!(
//                 "File extension of path {} is no valid Unicode string",
//                 file.display()
//             )
//         })?;
//         let extension_str_lower = extension_str.to_lowercase();
//         let factory = self
//             .writer_factories
//             .get(extension_str_lower.as_str())
//             .ok_or_else(|| {
//                 anyhow!(
//                     "Writing to point cloud files with extension {} is not supported",
//                     extension_str
//                 )
//             })?;

//         factory(file, point_layout)
//     }

//     /// Returns `true` if the associated `IOFactory` supports creating `PointReader` objects for the given
//     /// file `extension`
//     pub fn supports_reading_from(&self, extension: &str) -> bool {
//         let extension_lower = extension.to_lowercase();
//         self.reader_factories.contains_key(extension_lower.as_str())
//     }

//     /// Returns `true` if the associated `IOFactory` supports creating `PointWriter` objects for the given
//     /// file `extension`
//     pub fn supports_writing_to(&self, extension: &str) -> bool {
//         let extension_lower = extension.to_lowercase();
//         self.writer_factories.contains_key(extension_lower.as_str())
//     }

//     /// Register a new readable file extension with the associated `IOFactory`. The `reader_factory` will be called whenever
//     /// `extension` is encountered as a file extension in `make_reader`. Returns the previous reader factory function that
//     /// was registered for `extension`, if there was any. File extensions are treated as lower-case internally, so if the
//     /// extension `.FOO` is registered here, it will match `file.foo` and `file.FOO` (and all case-variations thereof).
//     pub fn register_reader_for_extension<
//         F: Fn(&Path) -> Result<Box<dyn PointReadAndSeek>> + 'static,
//     >(
//         &mut self,
//         extension: &str,
//         reader_factory: F,
//     ) -> Option<Box<ReaderFactoryFn>> {
//         let extension_lower = extension.to_lowercase();
//         self.reader_factories
//             .insert(extension_lower, Box::new(reader_factory))
//     }

//     /// Register a new writeable file extension with the associated `IOFactory`. The `writer_factory` will be called whenever
//     /// `extension` is encountered as a file extension in `make_writer`. Returns the previous writer factory function that
//     /// was registered for `extension`, if there was any. File extensions are treated as lower-case internally, so if the
//     /// extension `.FOO` is registered here, it will match `file.foo` and `file.FOO` (and all case-variations thereof).
//     pub fn register_writer_for_extension<
//         F: Fn(&Path, &PointLayout) -> Result<Box<dyn PointWriter>> + 'static,
//     >(
//         &mut self,
//         extension: &str,
//         writer_factory: F,
//     ) -> Option<Box<WriterFactoryFn>> {
//         let extension_lower = extension.to_lowercase();
//         self.writer_factories
//             .insert(extension_lower, Box::new(writer_factory))
//     }
// }

// impl Default for IOFactory {
//     fn default() -> Self {
//         let mut factory = Self {
//             reader_factories: Default::default(),
//             writer_factories: Default::default(),
//         };

//         factory.register_reader_for_extension("las", |path| {
//             let reader = LASReader::from_path(path)?;
//             Ok(Box::new(reader))
//         });
//         factory.register_writer_for_extension("las", |path, point_layout| {
//             let writer = LASWriter::from_path_and_point_layout(path, point_layout)?;
//             Ok(Box::new(writer))
//         });

//         factory.register_reader_for_extension("laz", |path| {
//             let reader = LASReader::from_path(path)?;
//             Ok(Box::new(reader))
//         });
//         factory.register_writer_for_extension("laz", |path, point_layout| {
//             let writer = LASWriter::from_path_and_point_layout(path, point_layout)?;
//             Ok(Box::new(writer))
//         });

//         factory
//     }
// }
