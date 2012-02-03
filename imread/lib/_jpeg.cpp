// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)
#include "base.h"
#include "_jpeg.h"

#include <stdio.h>
extern "C" {
#include <jpeglib.h>
}

namespace {
const size_t buffer_size = 4096;

struct jpeg_source_adaptor {
    jpeg_source_mgr mgr;
    byte_source* s;
    byte* buf;

    jpeg_source_adaptor(byte_source* s);

    ~jpeg_source_adaptor() {
        delete [] buf;
    }
};

struct jpeg_dst_adaptor {
    jpeg_destination_mgr mgr;
    byte_sink* s;
    byte* buf;

    jpeg_dst_adaptor(byte_sink* s);

    ~jpeg_dst_adaptor() {
        delete [] buf;
    }
};

void nop(j_decompress_ptr cinfo) { }
void nop_dst(j_compress_ptr cinfo) { }

boolean fill_input_buffer(j_decompress_ptr cinfo) {
    jpeg_source_adaptor* adaptor = reinterpret_cast<jpeg_source_adaptor*>(cinfo->src);
    adaptor->mgr.next_input_byte = adaptor->buf;
    adaptor->mgr.bytes_in_buffer = adaptor->s->read(adaptor->buf, buffer_size);
    return true;
}

void skip_input_data(j_decompress_ptr cinfo, long num_bytes) {
    if (num_bytes <= 0) return;
    jpeg_source_adaptor* adaptor = reinterpret_cast<jpeg_source_adaptor*>(cinfo->src);
    while (num_bytes > long(adaptor->mgr.bytes_in_buffer)) {
        num_bytes -= adaptor->mgr.bytes_in_buffer;
        fill_input_buffer(cinfo);
    }
    adaptor->mgr.next_input_byte += num_bytes;
    adaptor->mgr.bytes_in_buffer -= num_bytes;
}

boolean empty_output_buffer(j_compress_ptr cinfo) {
    jpeg_dst_adaptor* adaptor = reinterpret_cast<jpeg_dst_adaptor*>(cinfo->dest);
    adaptor->s->write(adaptor->buf, buffer_size);
    adaptor->mgr.next_output_byte = adaptor->buf;
    adaptor->mgr.free_in_buffer = buffer_size;
    return TRUE;
}

void flush_output_buffer(j_compress_ptr cinfo) {
    jpeg_dst_adaptor* adaptor = reinterpret_cast<jpeg_dst_adaptor*>(cinfo->dest);
    adaptor->s->write(adaptor->buf, adaptor->mgr.next_output_byte - adaptor->buf);
}



jpeg_source_adaptor::jpeg_source_adaptor(byte_source* s)
    :s(s)
    {
        buf = new byte[buffer_size];
        mgr.next_input_byte = buf;
        mgr.bytes_in_buffer = 0;

        mgr.init_source = nop;
        mgr.fill_input_buffer = fill_input_buffer;
        mgr.skip_input_data = skip_input_data;
        mgr.resync_to_restart = jpeg_resync_to_restart;
        mgr.term_source = nop;
    }

jpeg_dst_adaptor::jpeg_dst_adaptor(byte_sink* s)
    :s(s)
    {
        buf = new byte[buffer_size];

        mgr.next_output_byte = buf;
        mgr.free_in_buffer = buffer_size;

        mgr.init_destination = nop_dst;
        mgr.empty_output_buffer = empty_output_buffer;
        mgr.term_destination = flush_output_buffer;
    }
}

void JPEGFormat::read(byte_source* src, Image* output) {
    jpeg_source_adaptor adaptor(src);

    jpeg_decompress_struct cinfo;
    jpeg_create_decompress(&cinfo);

    // error management
    jpeg_error_mgr error;
    cinfo.err = jpeg_std_error(&error);

    // source
    cinfo.src = &adaptor.mgr;

    // now read the header & image data
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);
    const int h = cinfo.output_height;
    const int w = cinfo.output_width;
    const int d = cinfo.output_components;
    output->set_size(h, w, d);

    for (int r = 0; r != h; ++r) {
        byte* rowp = output->rowp_as<byte>(r);
        jpeg_read_scanlines(&cinfo, &rowp, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
}

void JPEGFormat::write(Image* input, byte_sink* output) {
    jpeg_dst_adaptor adaptor(output);

    jpeg_compress_struct cinfo;
    jpeg_create_compress(&cinfo);

    // error management
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    cinfo.err->trace_level = 128;

    cinfo.dest = &adaptor.mgr;

    cinfo.image_height = input->dim(0);
    cinfo.image_width = input->dim(1);
    cinfo.input_components = (input->ndims() > 2 ? input->dim(2) : 1);
    switch (cinfo.input_components) {
        case 1:
            cinfo.in_color_space = JCS_GRAYSCALE;
            break;
        case 3:
            cinfo.in_color_space = JCS_RGB;
            break;
        default:
            throw CannotWriteError("unsupported image dimensions");
    }

    jpeg_set_defaults(&cinfo);
    jpeg_start_compress(&cinfo, TRUE);

    while (cinfo.next_scanline < cinfo.image_height) {
        JSAMPROW rowp = static_cast<JSAMPROW>(input->rowp_as<void>(cinfo.next_scanline));
        (void) jpeg_write_scanlines(&cinfo, &rowp, 1);
    }
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
}


