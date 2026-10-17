# Welcome to imread's documentation!

Imread is a very simple library, with a small number of functions:

{func}`~imread.imread`
: Reads an image file

{func}`~imread.imread_multi`
: Reads all the images in a multi-image file (TIFF and STK)

{func}`~imread.imread_from_blob`
: Reads an image from a `bytes` object in memory

{func}`~imread.imsave`
: Writes an image file

{func}`~imread.imsave_multi`
: Writes several images into a single file (TIFF only)

{func}`~imread.detect_format`
: Detects the format of a file or blob from its magic number (PNG and JPEG
  only)

{func}`~imread.supports_format`
: Checks whether a format is supported

`imload`, `imload_multi`, `imload_from_blob`, `imwrite` and
`imwrite_multi` are aliases. The {mod}`imread.ijrois` module reads
[ImageJ](https://imagej.net/) regions of interest (see {doc}`formats`). All of
them are documented in the {doc}`api`.

It is meant to be used with a computer vision & image processing package, such
as:

- [mahotas](https://luispedro.org/software/mahotas)
- [scikit-image](https://scikit-image.org/)
- [OpenCV](https://opencv.org/)

This grew out of frustration at current image loading solutions in Python, in
either my packages [[mahotas](https://mahotas.rtfd.io)] or packages from
others [[scikit-image](https://scikit-image.org), for example].

The relationship with NumPy is very contained, so this could easily be
repurposed to load images in other frameworks, or even in other programming
languages.

## Citation

This package is an offshoot of mahotas. As it does not currently have its own
publication, you are asked to cite the parent package. If you use
mahotas-imread in a scientific publication, please cite:

> **Luis Pedro Coelho** Mahotas: Open source software for scriptable computer
> vision in Journal of Open Research Software, vol 1, 2013.
> [[DOI](https://doi.org/10.5334/jors.ac)]

In BibTeX format:

```bibtex
@article{mahotas,
    author = {Luis Pedro Coelho},
    title = {Mahotas: Open source software for scriptable computer vision},
    journal = {Journal of Open Research Software},
    year = {2013},
    doi = {10.5334/jors.ac},
    month = {July},
    volume = {1}
}
```

## Installation

The recommended way to install imread is with [pixi](https://pixi.sh):

```bash
pixi add imread
```

Binary wheels are also available with `pip install imread`. See
{doc}`install` for all the options, including building from source.

Contents:

```{toctree}
:maxdepth: 2

readme
install
formats
api
non-python
history
```

## Bug Reports

Please report any bugs either on [GitHub](https://github.com/luispedro/imread)
or by email to luis@luispedro.org.

If you have a test case where you are not sure whether imread is behaving
correctly, you can discuss it on the [pythonvision mailing list](https://groups.google.com/forum/?fromgroups#!forum/pythonvision).

If at all possible, include a small image as a test case.

# Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
